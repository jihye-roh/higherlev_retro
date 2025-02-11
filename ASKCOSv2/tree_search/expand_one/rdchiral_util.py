from rdchiral.initialization import rdchiralReaction, rdchiralReactants
from rdchiral.main import rdchiralRun
from rdkit import Chem
from typing import List, Tuple, Union


def bond_to_label(bond: Chem.Bond) -> str:
    """This function takes an RDKit bond and creates a label describing
    the most important attributes

    Args:
        bond (rdkit.Chem.rdchem.Bond): RDKit bond object

    Returns:
        str: String representing most important attributes of bond
    """

    a1_label = str(bond.GetBeginAtom().GetAtomicNum())
    a2_label = str(bond.GetEndAtom().GetAtomicNum())
    if bond.GetBeginAtom().GetAtomMapNum():
        a1_label += str(bond.GetBeginAtom().GetAtomMapNum())
    if bond.GetEndAtom().GetAtomMapNum():
        a2_label += str(bond.GetEndAtom().GetAtomMapNum())
    atoms = sorted([a1_label, a2_label])

    return f"{atoms[0]}{bond.GetSmarts()}{atoms[1]}"


def atoms_are_different(atom1: Chem.Atom, atom2: Chem.Atom) -> bool:
    """Compares two RDKit atoms based on basic properties

    Args:
        atom1 (rdkit.Chem.rdchem.Atom): First atom to compare
        atom2 (rdkit.Chem.rdchem.Atom): Second atom to compare

    Returns:
        bool: Whether the two atoms are different
    """

    if atom1.GetSmarts() != atom2.GetSmarts():
        return True  # should be very general
    if atom1.GetAtomicNum() != atom2.GetAtomicNum():
        return True  # must be true for atom mapping
    if atom1.GetTotalNumHs() != atom2.GetTotalNumHs():
        return True
    if atom1.GetFormalCharge() != atom2.GetFormalCharge():
        return True
    if atom1.GetDegree() != atom2.GetDegree():
        return True
    if atom1.GetNumRadicalElectrons() != atom2.GetNumRadicalElectrons():
        return True
    if atom1.GetIsAromatic() != atom2.GetIsAromatic():
        return True
    if atom1.GetIsotope() != atom2.GetIsotope():
        return True

    # Check bonds and nearest neighbor identity
    bonds1 = sorted([bond_to_label(bond) for bond in atom1.GetBonds()])
    bonds2 = sorted([bond_to_label(bond) for bond in atom2.GetBonds()])
    if bonds1 != bonds2:
        return True

    return False


def get_reacting_atoms(mapped_smiles: str) -> List[int]:
    reactants, _, product = mapped_smiles.split(">")
    reactants = Chem.MolFromSmiles(reactants)
    product = Chem.MolFromSmiles(product)
    atoms_r = {a.GetAtomMapNum(): a for a in reactants.GetAtoms() if a.GetAtomMapNum()}

    atoms_changed = []
    for atom_p in product.GetAtoms():
        x = atom_p.GetAtomMapNum()
        if not x:
            continue  # unnumbered
        try:
            atom_r = atoms_r[x]
        except KeyError:
            # a product atom unmapped in reactants indicates some atom mapping issue
            # just drop it (empty list is "ignored" by the frontend) to be conservative
            return []
            # continue  # unmapped

        if atoms_are_different(atom_r, atom_p):
            atoms_changed.append(x)

    return atoms_changed


def apply_one_template_to_precursors(precursors: str, template: str
                                     ) -> Tuple[dict, Union[str, None]]:
    """Apply one reversed retro template to precursors to get outcomes.
    Args:
        precursors (str): atom mapped smiles for precursors
        template (str): retro template to be applied

    Returns:
        mapped_products (dict): {smiles: mapped_smiles}
        mapped_precursors (str)
    """
    try:
        products, _, reactants = template.split(">")
        forward_template = f"({reactants})>>({products})"
        forward_rxn = rdchiralReaction(forward_template)
        precursor_reacts = rdchiralReactants(precursors)

        outcomes = rdchiralRun(forward_rxn, precursor_reacts, return_mapped=True)
    except Exception:
        return {}, None

    if outcomes:
        _, mapped_products = outcomes
        mapped_products = {k: v[0] for k, v in mapped_products.items()}
    else:
        mapped_products = {}

    try:
        mapped_precursors = precursor_reacts.smiles()  # rdchiral_cpp
    except AttributeError:
        # Python version of rdchiral
        mapped_precursors = Chem.MolToSmiles(precursor_reacts.reactants)

    return mapped_products, mapped_precursors
