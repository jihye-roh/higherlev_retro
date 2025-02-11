import argparse
import misc
import numpy as np
import os
import templ_rel_parser
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import utils
from dataset import FingerprintDataset, init_loader
from datetime import datetime
from tqdm import tqdm
from typing import Any, Dict, Tuple


def get_optimizer_and_scheduler(args, model: nn.Module, state: Dict[str, Any]) -> Tuple:
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="max",  # monitor top-1 val accuracy
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        cooldown=args.lr_cooldown,
        verbose=True,
    )

    if state and args.resume:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        misc.log_rank_0("Loaded pretrained optimizer and scheduler state_dicts.")

    return optimizer, scheduler


def _optimize(args, model: nn.Module, optimizer) -> float:
    nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
    optimizer.step()
    g_norm = utils.grad_norm(model)
    model.zero_grad(set_to_none=True)

    return g_norm


class TemplRelTrainer:
    """Class for Template Relevance Training"""

    def __init__(self, args):
        self.args = args

        self.model_name = args.model_name
        self.data_name = args.data_name
        self.log_file = args.log_file
        self.processed_data_path = args.processed_data_path
        self.model_path = args.model_path
        self.num_cores = args.num_cores

        self.model = None
        self.state = {}
        self.device = args.device

    def build_train_model(self) -> None:
        model, state = utils.get_model(self.args, self.device)
        misc.log_rank_0(model)
        misc.log_rank_0(f"Number of parameters = {utils.param_count(model)}")
        self.model = model
        self.state = state

    def train(self) -> None:
        # init optimizer and scheduler
        optimizer, scheduler = get_optimizer_and_scheduler(
            self.args, self.model, self.state
        )

        # init datasets and loaders
        train_dataset = FingerprintDataset(
            fp_file=os.path.join(self.processed_data_path, "product_fps_train.npz"),
            label_file=os.path.join(self.processed_data_path, "labels_train.npy"),
        )
        val_dataset = FingerprintDataset(
            fp_file=os.path.join(self.processed_data_path, "product_fps_val.npz"),
            label_file=os.path.join(self.processed_data_path, "labels_val.npy"),
        )

        train_loader = init_loader(
            args=self.args,
            dataset=train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
        )
        val_loader = init_loader(
            args=self.args,
            dataset=val_dataset,
            batch_size=self.args.val_batch_size,
            shuffle=False,
        )

        # final init
        max_val_acc = 0.0
        patience_counter = 0
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        start = time.time()
        misc.log_rank_0("Start training")
        for epoch in range(self.args.epochs):
            # training loop
            losses, accs = [], []
            train_loader = tqdm(train_loader, desc="training")
            self.model.train()
            self.model.zero_grad(set_to_none=True)

            for data in train_loader:
                inputs, labels = data
                inputs = inputs.to(self.device).float()
                labels = labels.to(self.device)

                logits = self.model(inputs).squeeze()
                loss, acc = self.model.get_loss(logits=logits, target=labels)
                loss.backward()
                losses.append(loss.item())
                accs.append(acc.item())

                _optimize(self.args, self.model, optimizer)

                train_loss = np.mean(losses)
                train_acc = np.mean(accs)

                train_loader.set_description(
                    f"training loss: {train_loss:.4f}, " f"top-1 acc: {train_acc:.4f}"
                )
                train_loader.refresh()

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            misc.log_rank_0(
                f"End of epoch {epoch}, "
                f"training loss: {train_loss:.4f}, "
                f"top-1 acc: {train_acc:.4f},"
                f"p_norm: {utils.param_norm(self.model):.4f}, "
                f"g_norm: {utils.grad_norm(self.model):.4f}, "
                f"lr: {utils.get_lr(optimizer):.6f}, "
                f"elapsed time: {time.time() - start:.0f}"
            )

            # validation loop (end of each epoch)
            self.model.eval()
            val_count = 128
            losses, accs = [], []
            val_loader = tqdm(val_loader, desc="validation")
            with torch.no_grad():
                for val_idx, data in enumerate(val_loader):
                    if val_idx >= val_count:
                        break
                    inputs, labels = data
                    inputs = inputs.to(self.device).float()
                    labels = labels.to(self.device)

                    logits = self.model(inputs).squeeze()
                    loss, acc = self.model.get_loss(logits=logits, target=labels)
                    losses.append(loss.item())
                    accs.append(acc.item())

                    val_loss = np.mean(losses)
                    val_acc = np.mean(accs)

                    val_loader.set_description(
                        f"validation loss: {val_loss:.4f}, "
                        f"top-1 acc: {val_acc:.4f}"
                    )
                    val_loader.refresh()

                val_losses.append(val_loss)
                val_accs.append(val_acc)
                misc.log_rank_0(
                    f"End of epoch {epoch}, "
                    f"avg. validation loss: {np.mean(val_losses)}, "
                    f"avg. val acc: {np.mean(val_accs): .4f},"
                    f"current val. loss: {val_loss},"
                    f"current val. acc: {val_acc: .4f},"
                    f"p_norm: {utils.param_norm(self.model): .4f}, "
                    f"g_norm: {utils.grad_norm(self.model): .4f}, "
                    f"lr: {utils.get_lr(optimizer): .6f}, "
                    f"elapsed time: {time.time() - start: .0f}\n"
                )
                self.model.train()
            
            # scheduler step
            scheduler.step(val_acc)
            misc.log_rank_0(
                f"Called a step of ReduceLROnPlateau,"
                f"current lr: {utils.get_lr(optimizer)}"
            )
            # Important: save only at one node for DDP or the ckpt would be corrupted!
            if dist.is_initialized() and dist.get_rank() > 0:
                continue

            # saving
            if val_acc > max_val_acc:
                misc.log_rank_0(f"Saving at the end of epoch {epoch}")
                state = {
                    "args": args,
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "train_losses": train_losses,
                    "train_accs": train_accs,
                    "val_losses": val_losses,
                    "val_accs": val_accs,
                    "max_val_acc": max_val_acc,
                }
                torch.save(
                    state, 
                    os.path.join(self.model_path, 
                    "model_latest.pt"))

            # early stopping
            if (
                self.args.early_stop
                and max_val_acc - val_acc >= self.args.early_stop_min_delta
            ):
                if patience_counter >= self.args.early_stop_patience:
                    misc.log_rank_0(f"Early stopped at the end of epoch: {epoch}")
                    break
                else:
                    patience_counter += 1
                    misc.log_rank_0(
                        f"Increase in val acc < early stop min delta "
                        f"{self.args.early_stop_min_delta}\n"
                        f"patience count: {patience_counter}."
                    )
            else:
                patience_counter = 0
                max_val_acc = max(max_val_acc, val_acc)

            # legacy, forcing synchronization
            if args.local_rank != -1:
                dist.barrier()


def train_main(args):
    args.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    utils.init_dist(args)
    misc.log_args(args, message="Logging training args")

    os.makedirs(args.model_path, exist_ok=True)
    trainer = TemplRelTrainer(args)
    trainer.build_train_model()
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("template_relevance")
    templ_rel_parser.add_model_opts(parser)
    templ_rel_parser.add_train_opts(parser)
    templ_rel_parser.add_predict_opts(parser)
    args, unknown = parser.parse_known_args()

    # logger setup
    os.makedirs("./logs/train", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")
    args.log_file = f"./logs/train/{args.log_file}.{dt}.log"
    logger = misc.setup_logger(args.log_file)

    utils.set_seed(args.seed)

    train_main(args)
