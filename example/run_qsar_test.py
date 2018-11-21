""""Module to to test the performance of the translation model to extract
    meaningfull features for a QSAR modelling. TWO QSAR datasets were extracted
    from literature:
    Ames mutagenicity: K. Hansen, S. Mika, T. Schroeter, A. Sutter, A. Ter Laak,
    T. Steger-Hartmann, N. Heinrich and K.-R. MuÌ´Lller, J. Chem.
    Inf. Model., 2009, 49, 2077–2081.
    Lipophilicity: Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse,
    A. S. Pappu, K. Leswing and V. Pande, Chemical Science, 2018,
    9, 513–530.
    """

import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.svm import SVC, SVR
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from cddd.inference import InferenceModel

FLAGS = None

def add_arguments(parser):
    """Helper function to fill the parser object.

    Args:
        parser: Parser object
    Returns:
        None
    """
    parser.add_argument('--model_path', default="default", type=str)
    parser.add_argument('--gpu', default=True, type=bool)
    parser.add_argument('--device', default="0", type=str)

def main(unused_argv):
    """Main function to test the performance of the translation model to extract
    meaningfull features for a QSAR modelling"""
    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.device)
    if FLAGS.model_path == "default":
        model_path = None
    else:
        model_path = FLAGS.model_path

    infer_model = InferenceModel(model_path)
    ames_df = pd.read_csv("ames.csv")
    ames_smls = ames_df.smiles.tolist()
    ames_labels = ames_df.label.values
    ames_fold = ames_df.fold.values
    ames_emb = infer_model.seq_to_emb(ames_smls)
    ames_emb = (ames_emb - ames_emb.mean()) / ames_emb.std()

    lipo_df = pd.read_csv("lipo.csv")
    lipo_smls = lipo_df.smiles.tolist()
    lipo_labels = lipo_df.label.values
    lipo_fold = lipo_df.fold.values
    lipo_emb = infer_model.seq_to_emb(lipo_smls)
    lipo_emb = (lipo_emb - lipo_emb.mean()) / lipo_emb.std()

    print("Running SVM on Ames mutagenicity...")
    clf = SVC(C=5.0)
    result = cross_val_score(clf,
                             ames_emb,
                             ames_labels,
                             ames_fold,
                             cv=LeaveOneGroupOut(),
                             n_jobs=5)
    print("Ames mutagenicity accuracy: %0.3f +/- %0.3f"
          %(np.mean(result), np.std(result)))

    print("Running SVM on Lipophilicity...")
    clf = SVR(C=5.0)
    result = cross_val_score(clf,
                             lipo_emb,
                             lipo_labels,
                             lipo_fold,
                             cv=LeaveOneGroupOut(),
                             n_jobs=5)
    print("Lipophilicity r2: %0.3f +/- %0.3f"
          %(np.mean(result), np.std(result)))

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    add_arguments(PARSER)
    FLAGS, UNPARSED = PARSER.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + UNPARSED)
