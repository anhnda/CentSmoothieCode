from rdkit import Chem
from rdkit.Chem import Draw
def plotX(sid,smile,highlight=None,highlightMap=None,withIndex = False):

    mol = Chem.MolFromSmiles(smile)

    def mol_with_atom_index(mol):
        atoms = mol.GetNumAtoms()
        for idx in range(atoms):
            mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))
        return mol
    if withIndex:
        mol = mol_with_atom_index(mol)
    Draw.MolToFile(mol,'../figs/%s.svg'%sid,highlightAtoms=highlight,highlightMap=highlightMap)