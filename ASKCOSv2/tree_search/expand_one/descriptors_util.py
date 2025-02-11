import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors


def molecular_weight(smiles: str) -> float:
    """
    Calculate exact molecular weight for the given SMILES string

    Args:
        smiles: SMILES string for which to calculate molecular weight

    Returns:
         float: exact molecular weight
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        molwt = Descriptors.ExactMolWt(mol)
    except Exception:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        mol.UpdatePropertyCache(strict=False)
        molwt = Descriptors.ExactMolWt(mol)

    return float(molwt)


def rms_molecular_weight(smiles: str) -> float:
    """Calculates the root-mean-square molecular weight for a given SMILES string

    Args:
        smiles: SMILES string for which to calculate root mean squared molecular weight

    Returns:
        float: root mean squared molecular weight

    """
    smiles_split = smiles.split(".")
    molwt_list = [molecular_weight(smi) for smi in smiles_split]
    rms_molwt = np.sqrt(np.mean(np.square(molwt_list)))

    return float(rms_molwt)


def number_of_rings(smiles: str) -> int:
    """Calculates the number of rings in a given SMILES string

    Args:
        smiles: SMILES string for which to calculate the number of rings

    Returns:
        int: number of rings

    """
    mol = Chem.MolFromSmiles(smiles)

    return int(mol.GetRingInfo().NumRings())
