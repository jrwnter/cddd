"""Functions to evaluate the performance of the translation model during training"""
import multiprocessing as mp
import csv
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.svm import SVC, SVR
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import scale
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score, auc
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
from cddd.inference import embedding2sequence, sequence2embedding

def parallel_eval_qsar(model, step, hparams):
    """Function that evaluates the translation models performance on one or multiple
    qsar-tasks at the current step. This is done in the background (training goes on)
    and multiple processes will spawn if eval frequnecy is low.

    Args:
        model: The model instance that should be evaluated.
        step: current step for logging.
        hparams: The hyperparameter object.
    Returns:
        A process object.
    Raises:
        ValueError: if input is not SMILES or INCHI.
    """
    df = pd.read_csv(hparams.infer_file)
    if "smiles" in hparams.input_sequence_key:
        if hparams.infer_input == "canonical":
            seq_list = df.canonical_smiles.tolist()
        elif hparams.infer_input == "random":
            seq_list = df.random_smiles.tolist()
    elif "inchi" in hparams.input_sequence_key:
        seq_list = df.inchi.tolist()
        seq_list = [seq.replace("InChI=1S", "") for seq in seq_list]
    else:
        raise ValueError("Could not understand the input typ. SMILES or INCHI?")
    label_array = df.label.values
    dataset_array = df.dataset.values
    fold_array = df.fold.values
    task_array = df.task.values
    embedding_array = sequence2embedding(model, hparams, seq_list)
    process = mp.Process(
        target=eval_qsar,
        args=(step, embedding_array, dataset_array, label_array, fold_array, task_array, hparams)
    )
    process.start()
    return process

def qsar_classification(emb, groups, labels):
    """Helper function that fits and scores a SVM classifier on the extracted molecular
    descriptor in a leave-one-group-out cross-validation manner.

    Args:
        emb: Embedding (molecular descriptor) that is used as input for the SVM
        groups: Array or list with n_samples entries defining the fold membership for the
        crossvalidtion.
        labels: Target values of the of the qsar task.
    Returns:
        The mean accuracy, F1-score, ROC-AUC and prescion-recall-AUC of the cross-validation.
    """
    acc = []
    f1 = []
    roc_auc = []
    pr_auc = []
    logo = LeaveOneGroupOut()
    clf = SVC(kernel='rbf', C=5.0, probability=True)
    for train_index, test_index in logo.split(emb, groups=groups):
        clf.fit(emb[train_index], labels[train_index])
        y_pred = clf.predict(emb[test_index])
        y_pred_prob = clf.predict_proba(emb[test_index])[:, 1]
        y_true = labels[test_index]
        precision, recall, t = precision_recall_curve(y_true, y_pred_prob)
        acc.append(accuracy_score(y_true, y_pred))
        f1.append(f1_score(y_true, y_pred))
        roc_auc.append(roc_auc_score(y_true, y_pred_prob))
        pr_auc.append(auc(recall, precision))
    return np.mean(acc), np.mean(f1), np.mean(roc_auc), np.mean(pr_auc)

def qsar_regression(emb, groups, labels):
    """Helper function that fits and scores a SVM regressor on the extracted molecular
    descriptor in a leave-one-group-out cross-validation manner.

    Args:
        emb: Embedding (molecular descriptor) that is used as input for the SVM
        groups: Array or list with n_samples entries defining the fold membership for the
        crossvalidtion.
        labels: Target values of the of the qsar task.
    Returns:
        The mean accuracy, F1-score, ROC-AUC and prescion-recall-AUC of the cross-validation.
    """
    r2 = []
    r = []
    mse = []
    mae = []
    logo = LeaveOneGroupOut()
    clf = SVR(kernel='rbf', C=5.0)
    for train_index, test_index in logo.split(emb, groups=groups):
        clf.fit(emb[train_index], labels[train_index])
        y_pred = clf.predict(emb[test_index])
        y_true = labels[test_index]
        r2.append(r2_score(y_true, y_pred))
        r.append(spearmanr(y_true, y_pred)[0])
        mse.append(mean_squared_error(y_true, y_pred))
        mae.append(mean_absolute_error(y_true, y_pred))
    return np.mean(r2), np.mean(r), np.mean(mse), np.mean(mae)

def eval_qsar(step, embedding_array, dataset_array, label_array, fold_array, task_array, hparams):
    """Function that runs a qsar experiment for multiple dataset and writes results to file.

    Args:
        step: current step for logging.
        embedding_array: The embedding (molecular descriptor) for the data (n_samples x n_features).
        dataset_array: Array with a dataset identifier (e.g. string) for each sample (n_samples).
        label_array: Target values of the of the qsar task(s) (n_samples).
        fold_array: Array with a fold membership identifier (int) for crossvalidtion (n_samples).
        task_array: Array with a task identifier (classification or regression)
        for each sample (n_samples).
        hparams: The hyperparameter object.
    Returns:
        None
    """
    header = ["step"]
    fields = [step]
    datasets = np.unique(dataset_array)
    for dataset in datasets:
        idxs = np.argwhere(dataset_array == dataset)[:, 0]
        emb = embedding_array[idxs]
        mean = np.mean(emb, axis=0)
        std = np.std(emb, axis=0)
        emb = (emb - mean) / std
        groups = fold_array[idxs]
        labels = label_array[idxs]
        if np.all(task_array[idxs] == "classification"):
            header += [dataset + "_" + measure for measure in ["accuracy",
                                                               "f1",
                                                               "roc_auc",
                                                               "pr_auc"]]
            measures = qsar_classification(emb, groups, labels)
        elif np.all(task_array[idxs] == "regression"):
            header += [dataset + "_" + measure for measure in ["r2", "spearman_r", "mse", "mae"]]
            measures = qsar_regression(emb, groups, labels)
        else:
            raise ValueError(
                "Not a conistent task specification (classification or regression) for ",
                dataset)

        fields.extend(list(measures))
    if step == 0:
        with open(os.path.join(hparams.save_dir, "eval_qsar.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    with open(os.path.join(hparams.save_dir, "eval_qsar.csv"), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

def eval_reconstruct(model, step, hparams):
    """Function that evaluates the translation models performance on character-wise translation
    accuracy and writes it to file.

    Args:
        model: The model instance that should be evaluated.
        step: current step for logging.
        hparams: The hyperparameter object.
    """
    header = ["step"] + list(model.model.measures_to_log.keys())
    fields = [step]
    if step == 0:
        with open(os.path.join(hparams.save_dir, "eval_reconstruct.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    measures = []
    while True:
        try:
            measures.append(model.model.eval(model.sess))
        except tf.errors.OutOfRangeError:
            break
    fields.extend(np.mean(measures, axis=0).tolist())
    with open(os.path.join(hparams.save_dir, "eval_reconstruct.csv"), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
