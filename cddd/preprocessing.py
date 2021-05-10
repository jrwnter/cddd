"""Functions that can be used to preprocess SMILES sequnces in the form used in the publication."""
import numpy as np
import pandas as pd
import tensorflow as tf
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit import Chem
from rdkit.Chem import Descriptors
REMOVER = SaltRemover()
ORGANIC_ATOM_SET = set([5, 6, 7, 8, 9, 15, 16, 17, 35, 53])


def dataframe_to_tfrecord(df,
                          tfrecord_file_name,
                          random_smiles_key=None,
                          canonical_smiles_key=None,
                          inchi_key=None,
                          mol_feature_keys=None,
                          shuffle_first=False):
    """Function to create a tf-record file to train the tranlation model from a pandas dataframe.
    Args:
        df: Dataframe with the sequnce representations of the molecules.
        tfrecord_file_name: Name/Path of the file to write the tf-record file to.
        random_smiles_key: header of the dataframe row which holds the randomized SMILES sequnces.
        canonical_smiles_key: header of the dataframe row which holds the canonicalized SMILES
        sequnces.
        inchi_key: header of the dataframe row which holds the InChI sequnces.
        mol_feature_keys:header of the dataframe row which holds molecualar features.
        shuffle_first: Defines if dataframe is shuffled first before writing to tf-record file.
    Returns:
        None
    """

    writer = tf.python_io.TFRecordWriter(tfrecord_file_name)
    if shuffle_first:
        df = df.sample(frac=1).reset_index(drop=True)
    for index, row in df.iterrows():
        feature_dict = {}
        if canonical_smiles_key is not None:
            canonical_smiles = row[canonical_smiles_key].encode("ascii")
            feature_dict["canonical_smiles"] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[canonical_smiles])
            )
        if random_smiles_key is not None:
            random_smiles = row[random_smiles_key].encode("ascii")
            feature_dict["random_smiles"] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[random_smiles])
            )
        if inchi_key is not None:
            inchi = row[inchi_key].encode("ascii")
            feature_dict["inchi"] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[inchi])
            )
        if mol_feature_keys is not None:
            mol_features = row[mol_feature_keys].values.astype(np.float32)
            feature_dict["mol_features"] = tf.train.Feature(
                float_list=tf.train.FloatList(value=mol_features)
            )
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()

def randomize_smile(sml):
    """Function that randomizes a SMILES sequnce. This was adapted from the
    implemetation of E. Bjerrum 2017, SMILES Enumeration as Data Augmentation
    for Neural Network Modeling of Molecules.
    Args:
        sml: SMILES sequnce to randomize.
    Return:
        randomized SMILES sequnce or
        nan if SMILES is not interpretable.
    """
    try:
        m = Chem.MolFromSmiles(sml)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m, ans)
        return Chem.MolToSmiles(nm, canonical=False)
    except:
        return float('nan')

def canonical_smile(sml):
    """Helper Function that returns the RDKit canonical SMILES for a input SMILES sequnce.
    Args:
        sml: SMILES sequence.
    Returns:
        canonical SMILES sequnce."""
    return Chem.MolToSmiles(sml, canonical=True)

def keep_largest_fragment(sml):
    """Function that returns the SMILES sequence of the largest fragment for a input
    SMILES sequnce.

    Args:
        sml: SMILES sequence.
    Returns:
        canonical SMILES sequnce of the largest fragment.
    """
    mol_frags = Chem.GetMolFrags(Chem.MolFromSmiles(sml), asMols=True)
    largest_mol = None
    largest_mol_size = 0
    for mol in mol_frags:
        size = mol.GetNumAtoms()
        if size > largest_mol_size:
            largest_mol = mol
            largest_mol_size = size
    return Chem.MolToSmiles(largest_mol)

def remove_salt_stereo(sml, remover):
    """Function that strips salts and removes stereochemistry information from a SMILES.
    Args:
        sml: SMILES sequence.
        remover: RDKit's SaltRemover object.
    Returns:
        canonical SMILES sequnce without salts and stereochemistry information.
    """
    try:
        sml = Chem.MolToSmiles(remover.StripMol(Chem.MolFromSmiles(sml),
                                                dontRemoveEverything=True),
                               isomericSmiles=False)
        if "." in sml:
            sml = keep_largest_fragment(sml)
    except:
        sml = np.float("nan")
    return sml

def organic_filter(sml):
    """Function that filters for organic molecules.
    Args:
        sml: SMILES sequence.
    Returns:
        True if sml can be interpreted by RDKit and is organic.
        False if sml cannot interpreted by RDKIT or is inorganic.
    """
    try:
        m = Chem.MolFromSmiles(sml)
        atom_num_list = [atom.GetAtomicNum() for atom in m.GetAtoms()]
        is_organic = (set(atom_num_list) <= ORGANIC_ATOM_SET)
        if is_organic:
            return sml
        else:
            return float('nan')
    except:
        return float('nan')

def filter_smiles(sml):
    try:
        m = Chem.MolFromSmiles(sml)
        logp = Descriptors.MolLogP(m)
        mol_weight = Descriptors.MolWt(m)
        num_heavy_atoms = Descriptors.HeavyAtomCount(m)
        if ((logp > -5) & (logp < 7) &
            (mol_weight > 12) & (mol_weight < 600) &
            (num_heavy_atoms > 3) & (num_heavy_atoms < 50)):
            return Chem.MolToSmiles(m)
        else:
            return float('nan')
    except:
        return float('nan')
    
def get_descriptors(sml):
    try:
        m = Chem.MolFromSmiles(sml)
        descriptor_list = []
        descriptor_list.append(Descriptors.MolLogP(m))
        descriptor_list.append(Descriptors.MolMR(m)) #ok
        descriptor_list.append(Descriptors.BalabanJ(m))
        descriptor_list.append(Descriptors.NumHAcceptors(m)) #ok
        descriptor_list.append(Descriptors.NumHDonors(m)) #ok
        descriptor_list.append(Descriptors.NumValenceElectrons(m))
        descriptor_list.append(Descriptors.TPSA(m)) # nice
        return descriptor_list
    except:
        return [np.float("nan")] * 7
def create_feature_df(smiles_df):
    temp = list(zip(*smiles_df['canonical_smiles'].map(get_descriptors)))
    columns = ["MolLogP", "MolMR", "BalabanJ", "NumHAcceptors", "NumHDonors", "NumValenceElectrons", "TPSA"]
    df = pd.DataFrame(columns=columns)
    for i, c in enumerate(columns):
        df.loc[:, c] = temp[i]
    df = (df - df.mean(axis=0, numeric_only=True)) / df.std(axis=0, numeric_only=True)
    df = smiles_df.join(df)
    return df

def preprocess_smiles(sml):
    """Function that preprocesses a SMILES string such that it is in the same format as
    the translation model was trained on. It removes salts and stereochemistry from the
    SMILES sequnce. If the sequnce correspond to an inorganic molecule or cannot be
    interpreted by RDKit nan is returned.

    Args:
        sml: SMILES sequence.
    Returns:
        preprocessd SMILES sequnces or nan.
    """
    new_sml = remove_salt_stereo(sml, REMOVER)
    new_sml = filter_smiles(new_sml)
    return new_sml


def preprocess_list(smiles):
    df = pd.DataFrame(smiles)
    df["canonical_smiles"] = df[0].map(preprocess_smiles)
    df = df.drop([0], axis=1)
    df = df.dropna(subset=["canonical_smiles"])
    df = df.reset_index(drop=True)
    df["random_smiles"] = df["canonical_smiles"].map(randomize_smile)
    df = create_feature_df(df)
    return df
    