import numpy as np
import pandas as pd
import multiprocessing as mp
import csv
import os
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import scale
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score, auc, accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
from .input_pipeline import InputPipelineInfer


def parallel_inference(model, step, hparams):
    emb_list = []
    df = pd.read_csv(hparams.infer_file)
    if (hparams.input_pipeline == "SmlToCanSml") | (hparams.input_pipeline == "CanSmlToCanSml") | (hparams.input_pipeline == "CanSmlToInchi") | (hparams.input_pipeline == "SmlToCanSmlWithFeatures") | (hparams.input_pipeline == "CanSmlToCanSmlWithFeatures"):
        if hparams.infer_input == "canonical":
            seq_list = df.canonical_smiles.tolist()
        elif hparams.infer_input == "random":
            seq_list = df.random_smiles.tolist()
    elif (hparams.input_pipeline == "InchiToCanSml") | (hparams.input_pipeline =="InchiToCanSmlWithFeatures"):
        seq_list = df.inchi.tolist()
        seq_list = [seq.replace("InChI=1S", "") for seq in seq_list]
    else:
        raise ValueError("Inference method doesn't know this input_pipeline...")
    label_array = df.label.as_matrix()
    dataset_array = df.dataset.as_matrix()
    fold_array = df.fold.as_matrix()
    with model.graph.as_default():
        input_pipeline = InputPipelineInfer(seq_list, hparams)
        input_pipeline.initilize()
        model.model.restore(model.sess)
        while 1:
            try:
                input_seq, input_len = input_pipeline.get_next()
                emb = model.model.get_embedding(model.sess, input_seq, input_len)
                emb_list.append(emb)
            except StopIteration:
                break
        embedding_array = np.concatenate(emb_list)
        process = mp.Process(target=full_inference, args=(step, embedding_array, dataset_array, label_array, fold_array, hparams))
        process.start()
    return process

def simple_inference(model, hparams, seq_list=None):
    emb_list = []
    if seq_list is None:
        df = pd.read_csv(hparams.infer_file)
        if (hparams.input_pipeline == "SmlToCanSml") | (hparams.input_pipeline == "CanSmlToInchi") | (hparams.input_pipeline == "SmlToCanSmlWithFeatures"):
            if hparams.infer_input == "canonical":
                seq_list = df.canonical_smiles.tolist()
            elif hparams.infer_input == "random":
                seq_list = df.random_smiles.tolist()
        elif hparams.input_pipeline == "InchiToCanSml":
            seq_list = df.inchi.tolist()
            seq_list = [seq.replace("InChI=1S", "") for seq in seq_list]
    with model.graph.as_default():
        input_pipeline = InputPipelineInfer(seq_list, hparams)
        input_pipeline.initilize()
        model.model.restore(model.sess)
        while 1:
            try:
                input_seq, input_len = input_pipeline.get_next()
                emb = model.model.get_embedding(model.sess, input_seq, input_len)
                emb_list.append(emb)
            except StopIteration:
                break
        embedding_array = np.concatenate(emb_list)
    return embedding_array

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def simple_inference_decode(model, hparams, embedding, num_top=1):
    seq_list = []
    with model.graph.as_default():
        input_pipeline = InputPipelineInfer(seq_list, hparams)
        input_pipeline.initilize()
        model.model.restore(model.sess)
        for x in batch(embedding, model.model.batch_size):
            seq = model.model.decode_from_embedding(model.sess, x, num_top)
            if num_top == 1:
                seq = [s[0] for s in seq]
            seq_list.extend(seq)
        if (len(seq_list) == 1 ) & isinstance(seq_list, str):
            return seq_list[0]
    return seq_list


def get_embedding(model, sess, input_pipeline, hparams):
    emb_list = []
    while 1:
        try:
            input_seq, input_len = input_pipeline.get_next()
            emb = model.get_embedding(sess, input_seq, input_len)
            emb_list.append(emb)
        except StopIteration:
            break
    embedding_array = np.concatenate(emb_list)
    return embedding_array

def full_inference(step, embedding_array, dataset_array, label_array, fold_array, hparams):
    summary = []
    fields = [step]
    logo = LeaveOneGroupOut()
    for dataset in ["ames", "lipo"]:
        idxs = np.argwhere(dataset_array == dataset)[:, 0]
        measure1 = []
        measure2 = []
        measure3 = []
        measure4 = []
        if dataset in ["ames"]:
            clf = SVC(kernel='rbf', C=5.0, probability=True)
            emb = scale(embedding_array[idxs])
            groups = fold_array[idxs]
            labels = label_array[idxs]
            for train_index, test_index in logo.split(emb, groups=groups):
                clf.fit(emb[train_index], labels[train_index])
                y_pred = clf.predict(emb[test_index])
                y_pred_prob = clf.predict_proba(emb[test_index])[:, 1]
                y_true = labels[test_index]
                precision, recall, t = precision_recall_curve(y_true, y_pred_prob)
                measure1.append(accuracy_score(y_true, y_pred))
                measure2.append(f1_score(y_true, y_pred))
                measure3.append(roc_auc_score(y_true, y_pred_prob))
                measure4.append(auc(recall, precision))
        else:
            clf = SVR(kernel='rbf', C=5.0)
            emb = scale(embedding_array[idxs])
            groups = fold_array[idxs]
            labels = label_array[idxs]
            for train_index, test_index in logo.split(emb, groups=groups):
                clf.fit(emb[train_index], labels[train_index])
                y_pred = clf.predict(emb[test_index])
                y_true = labels[test_index]
                measure1.append(r2_score(y_true, y_pred))
                measure2.append(spearmanr(y_true, y_pred)[0])
                measure3.append(mean_squared_error(y_true, y_pred))
                measure4.append(mean_absolute_error(y_true, y_pred))

        fields.extend([np.mean(measure1), np.mean(measure2), np.mean(measure3), np.mean(measure4)])
    if step == 0:
        header = ["step"]
        header += [dataset + "_" + measure for dataset in ["ames"] for measure in ["accuracy", "f1", "roc_auc", "pr_auc"]]
        header += [dataset + "_" + measure for dataset in ["lipo"] for measure in ["r2", "spearman_r", "mse", "mae"]]
        with open(os.path.join(hparams.save_dir, "INFER.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
    with open(os.path.join(hparams.save_dir, "INFER.csv"), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

