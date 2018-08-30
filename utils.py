import tensorflow as tf
import numpy as np
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit import Chem
def grad_summary(grads):
    """
    Helper function to receive a summary for the input gradiens
    Args:
        grads: gradients
    Returns:
        grad_summary: tensorflow summary of the gradients
    """
    grad_sum = []
    for grad, var in grads:
        if grad is not None:
            grad_sum.append(tf.summary.histogram(var.op.name + '/grads', grad))
    return grad_sum

def save_output(arr, file_name):
    if file_name.endswith(".csv"):
        np.savetxt(file_name, arr, delimiter=",")
    elif file_name.endswith(".tab"):
        np.savetxt(file_name, arr, delimiter="\t")
    elif file_name.endswith(".npy"):
        np.save(file_name, arr)
    else:
        raise ValueError("Choose one of following file types: .csv, .tab, .npy")
        
        

remover = SaltRemover()
ORGANIC_ATOM_SET = set([5, 6, 7, 8, 9, 15, 16, 17, 35, 53])
def remove_salt(sml, remover):
    try:
        sml = Chem.MolToSmiles(remover.StripMol(Chem.MolFromSmiles(sml), dontRemoveEverything=True))
        if "." in sml:
            sml = np.float("nan")
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
def preprocess_smiles(sml_list):
    sml_list = list(set(sml_list))
    old_to_new_sml_dic = {}
    for old_sml in sml_list:
        new_sml = remove_salt(old_sml, remover)
        if new_sml != np.float("nan"):
            if not organic_filter(new_sml):
                new_sml = np.float("nan")
        old_to_new_sml_dic[old_sml] = new_sml
    return old_to_new_sml_dic

def preprocess_smiles2(sml):
    new_sml = remove_salt(sml, remover)
    if new_sml != np.float("nan"):
        if not organic_filter(new_sml):
            new_sml = np.float("nan")
    return new_sml
    