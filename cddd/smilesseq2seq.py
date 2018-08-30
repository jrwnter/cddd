import tensorflow as tf
import numpy as np
import os
import sys
import shutil
import pandas as pd
import time
import argparse
import json
from seq2seq.input_pipeline import *
from seq2seq.model_helper import build_models
from seq2seq.inference import parallel_inference
from seq2seq.evaluation import write_to_logfile
from seq2seq.utils import save_output
tf.logging.set_verbosity(tf.logging.ERROR)
FLAGS = None


def add_arguments(parser):
    parser.add_argument('-m', '--model', help='which model?', default="GRUSeq2Seq")
    parser.add_argument('-i', '--input_pipeline', default="SmlToCanSml")
    parser.add_argument('-c','--cell_size',
                        help='hidden layers of cell. multiple comma seperated numbers for for multi layer rnn',
                        nargs='+', default=[128], type=int)
    parser.add_argument('-e','--emb_size', help='size of bottleneck layer', default=128, type=int)
    parser.add_argument('-l','--learning_rate', default=0.0005, type=int)
    parser.add_argument('-s','--save_dir', help='path to save and log files', default=".", type=str)
    parser.add_argument('-d','--device', help="number of cuda visible devise", default=-1, type=str)
    parser.add_argument('-r', '--restore', help="restore oldd model?", default=False, type=bool)
    parser.add_argument('-gmf', '--gpu_mem_frac', default=1.0, type=float)
    parser.add_argument('-n','--num_steps', help="number of train steps", default=250000, type=int)
    parser.add_argument('--summary_freq', default=1000, type=int)
    parser.add_argument('--inference_freq', default=5000, type=int)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--char_embedding_size', default=32)
    parser.add_argument('--encode_vocabulary_file', default="indices_char.npy", type=str)
    parser.add_argument('--decode_vocabulary_file', default="indices_char.npy", type=str)
    parser.add_argument('--train_file', default="../data/pretrain_dataset.tfrecords", type=str)
    parser.add_argument('--val_file', default="../data/pretrain_dataset_val.tfrecords", type=str)
    parser.add_argument('--infer_file', default="../data/val_infer.csv", type=str)
    parser.add_argument('--allow_soft_placement', default=True)
    parser.add_argument('--cpu_threads', default=5)
    parser.add_argument('--overwrite_saves', default=False)
    parser.add_argument('--input_dropout', default=0.0, type=float)
    parser.add_argument('--emb_noise', default=0.0, type=float)
    parser.add_argument('-ks','--kernel_size', nargs='+', default=[2], type=int)
    parser.add_argument('-chs', '--conv_hidden_size', nargs='+', default=[128], type=int)
    parser.add_argument('--reverse_decoding', default=False, type=bool)
    parser.add_argument('--buffer_size', default=10000, type=int)
    parser.add_argument('--lr_decay', default=True, type=bool)
    parser.add_argument('--lr_decay_frequency', default=50000, type=int)
    parser.add_argument('--lr_decay_factor', default=0.9, type=int)
    parser.add_argument('--num_buckets', default=8., type=float)
    parser.add_argument('--min_bucket_length', default=20.0, type=float)
    parser.add_argument('--max_bucket_length', default=60.0, type=float)
    parser.add_argument('--num_features', default=7, type=int)
    parser.add_argument('--only_infer', default=False, type=bool)
    parser.add_argument('--save_hparams', default=True, type=bool)
    parser.add_argument('--hparams_from_file', default=False, type=bool)
    parser.add_argument('--hparams_file_name', default=None, type=str)
    parser.add_argument('--output_file_name', default=None, type=str)
    parser.add_argument('--rand_input_swap', default=False, type=bool)
    parser.add_argument('--infer_input', default="random", type=str)
    
    
    
def create_hparams(flags):
    """Create training hparams."""
    hparams = tf.contrib.training.HParams(
        model = flags.model,
        input_pipeline = flags.input_pipeline,
        cell_size = flags.cell_size,
        emb_size = flags.emb_size,
        save_dir = flags.save_dir,
        device = flags.device,
        restore = flags.restore,
        lr = flags.learning_rate,
        gpu_mem_frac = flags.gpu_mem_frac,
        num_steps = flags.num_steps,
        summary_freq = flags.summary_freq,
        inference_freq = flags.inference_freq,
        batch_size = flags.batch_size,
        char_embedding_size = flags.char_embedding_size,
        encode_vocabulary_file = flags.encode_vocabulary_file,
        decode_vocabulary_file = flags.decode_vocabulary_file,
        train_file = flags.train_file,
        val_file = flags.val_file,
        infer_file = flags.infer_file,
        allow_soft_placement = flags.allow_soft_placement,
        cpu_threads = flags.cpu_threads,
        overwrite_saves = flags.overwrite_saves,
        input_dropout = flags.input_dropout,
        emb_noise = flags.emb_noise,
        conv_hidden_size = flags.conv_hidden_size,
        kernel_size = flags.kernel_size,
        reverse_decoding = flags.reverse_decoding,
        buffer_size = flags.buffer_size,
        lr_decay = flags.lr_decay,
        lr_decay_frequency = flags.lr_decay_frequency,
        lr_decay_factor = flags.lr_decay_factor,
        num_buckets = flags.num_buckets,
        min_bucket_length = flags.min_bucket_length,
        max_bucket_length = flags.max_bucket_length,
        num_features = flags.num_features,
        output_file_name = flags.output_file_name,
        rand_input_swap = flags.rand_input_swap,
        infer_input = flags.infer_input,
    )
    hparams_file_name = flags.hparams_file_name
    if hparams_file_name is None:
        hparams_file_name = os.path.join(hparams.save_dir, 'hparams.json')
    if flags.hparams_from_file:
        hparams.cell_size = list()
        hparams = hparams.parse_json(json.load(open(hparams_file_name)))
    return hparams

def train_loop(train_model, eval_model, infer_model, hparams):
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
                write_to_logfile(eval_model, step, hparams)
        if step % hparams.inference_freq == 0:
            with infer_model.graph.as_default():
                infer_process.append(parallel_inference(infer_model, step, hparams))
    for p in infer_process:
        p.join()
    if hparams.output_file_name is not None:
        inference_run(infer_model, hparams)
    
def inference_run(infer_model, hparams):
    with infer_model.graph.as_default():
        embedding_array = simple_inference(infer_model, hparams)
        save_output(embedding_array, hparams.output_file_name)
    
    
def main(unused_argv):
    hparams = create_hparams(FLAGS)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(hparams.device)
    if FLAGS.only_infer:
        infer_model = build_models(hparams, only_infer=True)
        inference_run(infer_model)
    else:
        train_model, eval_model, infer_model = build_models(hparams)
        train_loop(train_model, eval_model, infer_model, hparams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    