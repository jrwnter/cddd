import tensorflow as tf
import numpy as np
import os
import sys
import shutil
import pandas as pd
import time
import argparse
import json
from cddd.model_helper import build_models
from cddd.evaluation import eval_reconstruct, parallel_eval_qsar
from cdd.hyperparameters import add_arguments, create_hparams
tf.logging.set_verbosity(tf.logging.ERROR)
FLAGS = None

def train_loop(train_model, eval_model, encoder_model, hparams):
    infer_process = []
    with train_model.graph.as_default():
        if hparams.restore:
            step = train_model.model.restore(train_model.sess)
        else:
            train_model.sess.run(train_model.model.iterator.initializer)
            step = train_model.model.initilize(train_model.sess, overwrite_saves=hparams.overwrite_saves)
    hparams_file_name = FLAGS.hparams_file_name
    if hparams_file_name is None:
        hparams_file_name = os.path.join(hparams.save_dir, 'hparams.json')
    with open(hparams_file_name, 'w') as outfile:
        json.dump(hparams.to_json(), outfile)
    while step < hparams.num_steps:
        with train_model.graph.as_default():
            step = train_model.model.train(train_model.sess)
        if step % hparams.summary_freq == 0:
            with train_model.graph.as_default():
                train_model.model.save(train_model.sess)
            with eval_model.graph.as_default():
                eval_model.model.restore(eval_model.sess)
                eval_model.sess.run(eval_model.model.iterator.initializer)
                eval_reconstruct(eval_model, step, hparams)
        if step % hparams.inference_freq == 0:
            with encoder_model.graph.as_default():
                infer_process.append(parallel_eval_qsar(encoder_model, step, hparams))
    for p in infer_process:
        p.join()
    
def main(unused_argv):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(hparams.device)
    train_model, eval_model, encode_model = build_models(hparams)
    train_loop(train_model, eval_model, encode_model, hparams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    