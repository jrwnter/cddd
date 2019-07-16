"""Main module for training the translation model to learn extracting CDDDs"""
import os
import sys
import argparse
import json
import tensorflow as tf
from cddd.model_helper import build_models
from cddd.evaluation import eval_reconstruct, parallel_eval_qsar
from cddd.hyperparameters import add_arguments, create_hparams

tf.logging.set_verbosity(tf.logging.ERROR)
FLAGS = None

def train_loop(train_model, eval_model, encoder_model, hparams):
    """Main training loop function for training and evaluating.
    Args:
        train_model: The model used for training.
        eval_model: The model used evaluating the translation accuracy.
        encoder_model: The model used for evaluating the QSAR modeling performance.
        hparams: Hyperparameters defined in file or flags.
    Returns:
        None
    """
    qsar_process = []
    with train_model.graph.as_default():
        train_model.sess.run(train_model.model.iterator.initializer)
        step = train_model.model.initilize(
            train_model.sess,
            overwrite_saves=hparams.overwrite_saves
        )
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
                qsar_process.append(parallel_eval_qsar(encoder_model,
                                                       step,
                                                       hparams))
    for process in qsar_process:
        process.join()

def main(unused_argv):
    """Main function that trains and evaluats the translation model"""
    hparams = create_hparams(FLAGS)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(hparams.device)
    train_model, eval_model, encode_model = build_models(hparams)
    train_loop(train_model, eval_model, encode_model, hparams)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    add_arguments(PARSER)
    FLAGS, UNPARSED = PARSER.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + UNPARSED)
    