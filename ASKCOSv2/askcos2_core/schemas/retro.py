from pydantic import Field, validator
from schemas.base import LowerCamelAliasModel
from typing import Any, Literal


class RetroBackendOption(LowerCamelAliasModel):
    retro_backend: Literal[
        "augmented_transformer", "graph2smiles", "template_relevance", "retrosim"
    ] = Field(
        default="template_relevance",
        description="backend for one-step retrosynthesis"
    )
    retro_model_name: str = Field(
        default="reaxys",
        description="backend model name for one-step retrosynthesis"
    )

    max_num_templates: int = Field(
        default=1000,
        description="number of templates to consider"
    )
    max_cum_prob: float = Field(
        default=0.995,
        description="maximum cumulative probability of templates"
    )
    attribute_filter: list[dict[str, Any]] = Field(
        default_factory=list,
        description="template attribute filter to apply before template application",
        example=[]
    )
    threshold: float = Field(
        default=0.3,
        description="threshold for similarity; "
                    "used for retrosim only"
    )
    top_k: int = Field(
        default=10,
        description="filter for k results returned; "
                    "used for retrosim only"
    )

    @validator("retro_model_name")
    def check_retro_model_name(cls, v, values):
        if "retro_backend" not in values:
            raise ValueError("retro_backend not supplied!")
        if values["retro_backend"] == "template_relevance": return v
    
        elif values["retro_backend"] == "augmented_transformer":
            if v not in [
                "cas",
                "pistachio_23Q3",
                "USPTO_FULL"
            ]:
                raise ValueError(
                    f"Unsupported retro_model_name {v} for augmented_transformer")
        elif values["retro_backend"] == "graph2smiles":
            if v not in [
                "cas",
                "pistachio_23Q3",
                "USPTO_FULL"
            ]:
                raise ValueError(
                    f"Unsupported retro_model_name {v} for graph2smiles")
        elif values["retro_backend"] == "retrosim":
            if v not in [
                "USPTO_FULL",
                "bkms"
            ]:
                raise ValueError(
                    f"Unsupported retro_model_name {v} for retrosim")
        return v
