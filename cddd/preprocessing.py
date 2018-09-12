import pandas as pd
import tensorflow as tf
import numpy as np
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit import Chem  
remover = SaltRemover()
ORGANIC_ATOM_SET = set([5, 6, 7, 8, 9, 15, 16, 17, 35, 53])


def dataframe_to_tfrecord(df, tfrecord_file_name, random_smiles_key=None, canonical_smiles_key=None, inchi_key=None, mol_feature_keys=None, shuffle_first=False):
    writer = tf.python_io.TFRecordWriter(tfrecord_file_name)
    if shuffle_first:
        df = df.sample(frac=1).reset_index(drop=True)
    for index, row in df.iterrows():
        feature_dict = {}
        if canonical_smiles_key is not None:
            canonical_smiles = row[canonical_smiles_key].encode("ascii")
            feature_dict["canonical_smiles"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[canonical_smiles]))
        if random_smiles_key is not None:
            random_smiles = row[random_smiles_key].encode("ascii")
            feature_dict["random_smiles"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[random_smiles]))
        if inchi_key is not None:
            inchi = row[inchi_key].encode("ascii")
            feature_dict["inchi"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[inchi]))
        if mol_feature_keys is not None:
            mol_features = row[mol_feature_keys].values.astype(np.float32)
            feature_dict["mol_features"] = tf.train.Feature(float_list=tf.train.FloatList(value=mol_features))
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()
    
def randomize_smile(sml):
    """randomize a SMILES
       This was adapted from the implemetation of E. Bjerrum 2017, 
       SMILES Enumeration as Data Augmentation for Neural Network Modeling
       of Molecules"""
    try:
        m = Chem.MolFromSmiles(sml)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m, ans)
        return Chem.MolToSmiles(nm, canonical=False)
    except:
        return float('nan')
    
def canonical_smile(sml):
    return Chem.MolToSmiles(sml, canonical=True)

def keep_largest_fragment(sml):
    mol_frags = Chem.GetMolFrags(Chem.MolFromSmiles(sml), asMols=True)
    largest_mol = None
    largest_mol_size = 0
    for mol in mol_frags:
        size = mol.GetNumAtoms()
        if size > largest_mol_size:
            largest_mol = mol
            largest_mol_size = size
    return Chem.MolToSmiles(largest_mol)
     
def remove_salt(sml, remover):
    try:
        sml = Chem.MolToSmiles(remover.StripMol(Chem.MolFromSmiles(sml), dontRemoveEverything=True))
        if "." in sml:
            sml = keep_largest_fragment(sml)
    except:
        sml = np.float("nan")
    return(sml)
def organic_filter(sml):
    try:
        m = Chem.MolFromSmiles(sml)
        atom_num_list = [atom.GetAtomicNum() for atom in m.GetAtoms()]
        is_organic = (set(atom_num_list) <= ORGANIC_ATOM_SET)
        if is_organic:
            return True
        else:
            return False
    except:
        return False

def preprocess_smiles(sml):
    new_sml = remove_salt(sml, remover)
    if new_sml != np.float("nan"):
        if not organic_filter(new_sml):
            new_sml = np.float("nan")
    return new_sml
    