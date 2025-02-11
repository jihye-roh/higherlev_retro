from rdkit import Chem


def pass_bond_edits_test(r: str, p: str, max_rbonds=5, max_pbonds=3, max_atoms=10):
    """Adapted from filter.py and rdkit.py in temprel"""
    rmol = Chem.MolFromSmiles(r)
    pmol = Chem.MolFromSmiles(p)

    if not rmol or not pmol:
        return False

    pbonds = []
    for bond in pmol.GetBonds():
        a = bond.GetBeginAtom().GetAtomMapNum()
        b = bond.GetEndAtom().GetAtomMapNum()
        if a or b:
            pbonds.append(tuple(sorted([a, b])))

    rbonds = []
    for bond in rmol.GetBonds():
        a = bond.GetBeginAtom().GetAtomMapNum()
        b = bond.GetEndAtom().GetAtomMapNum()
        if a or b:
            rbonds.append(tuple(sorted([a, b])))

    r_changed = set(rbonds) - set(pbonds)
    p_changed = set(pbonds) - set(rbonds)

    if len(r_changed) > max_rbonds or len(p_changed) > max_pbonds:
        return False

    atoms_changed = set()
    for ch in list(r_changed) + list(p_changed):
        atoms_changed.add(ch[0])
        atoms_changed.add(ch[1])
    atoms_changed -= {0}

    if len(atoms_changed) > max_atoms:
        return False

    # if passing all three criteria
    return True
