"""Helper functions to parse and create a hyperparameter object."""

import os
import json
import tensorflow as tf
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

def add_arguments(parser):
    """Helper function to fill the parser object.

    Args:
        parser: Parser object
    Returns:
        None
    """
    parser.add_argument('-m', '--model', help='which model?',
                        default="NoisyGRUSeq2SeqWithFeatures", type=str)
    parser.add_argument('-i', '--input_pipeline', default="InputPipelineWithFeatures", type=str)
    parser.add_argument('--input_sequence_key', default="random_smiles", type=str)
    parser.add_argument('--output_sequence_key', default="canonical_smiles", type=str)
    parser.add_argument('-c', '--cell_size',
                        help='hidden layers of cell. multiple numbers for multi layer rnn',
                        nargs='+',
                        default=[128],
                        type=int)
    parser.add_argument('-e', '--emb_size', help='size of bottleneck layer', default=128, type=int)
    parser.add_argument('-l', '--learning_rate', default=0.0005, type=float)
    parser.add_argument('-s', '--save_dir',
                        help='path to save and log files', default=os.path.join(DEFAULT_DATA_DIR, 'default_model'), type=str)
    parser.add_argument('-d', '--device',
                        help="number of cuda visible devise", default="-1", type=str)
    parser.add_argument('-gmf', '--gpu_mem_frac', default=1.0, type=float)
    parser.add_argument('-n', '--num_steps', help="number of train steps", default=250000, type=int)
    parser.add_argument('--summary_freq', help='save model and log translation accuracy',
                        default=1000, type=int)
    parser.add_argument('--inference_freq', help='log qsar modelling performance', default=5000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--one_hot_embedding', default=False, type=bool)
    parser.add_argument('--char_embedding_size', default=32, type=int)
    parser.add_argument('--train_file', default="../data/pretrain_dataset.tfrecords", type=str)
    parser.add_argument('--val_file', default="../data/pretrain_dataset_val.tfrecords", type=str)
    parser.add_argument('--infer_file', default="../data/val_dataset_preprocessed3.csv", type=str)
    parser.add_argument('--allow_soft_placement', default=True, type=bool)
    parser.add_argument('--cpu_threads', default=5, type=int)
    parser.add_argument('--overwrite_saves', default=False, type=bool)
    parser.add_argument('--input_dropout', default=0.0, type=float)
    parser.add_argument('--emb_noise', default=0.0, type=float)
    parser.add_argument('-ks', '--kernel_size', nargs='+', default=[2], type=int)
    parser.add_argument('-chs', '--conv_hidden_size', nargs='+', default=[128], type=int)
    parser.add_argument('--reverse_decoding', default=False, type=bool)
    parser.add_argument('--buffer_size', default=10000, type=int)
    parser.add_argument('--lr_decay', default=True, type=bool)
    parser.add_argument('--lr_decay_frequency', default=50000, type=int)
    parser.add_argument('--lr_decay_factor', default=0.9, type=float)
    parser.add_argument('--num_buckets', default=8., type=float)
    parser.add_argument('--min_bucket_length', default=20.0, type=float)
    parser.add_argument('--max_bucket_length', default=60.0, type=float)
    parser.add_argument('--num_features', default=7, type=int)
    parser.add_argument('--save_hparams', default=True, type=bool)
    parser.add_argument('--hparams_from_file', default=False, type=bool)
    parser.add_argument('--hparams_file_name', default=None, type=str)
    parser.add_argument('--rand_input_swap', default=False, type=bool)
    parser.add_argument('--infer_input', default="random", type=str)
    parser.add_argument('--emb_activation', default="tanh", type=str)
    parser.add_argument('--div_loss_scale', default=1.0, type=float)
    parser.add_argument('--div_loss_rate', default=0.9, type=float)

def create_hparams(flags):
    """Create training hparams."""
    hparams = tf.contrib.training.HParams(
        model=flags.model,
        input_pipeline=flags.input_pipeline,
        input_sequence_key=flags.input_sequence_key,
        output_sequence_key=flags.output_sequence_key,
        cell_size=flags.cell_size,
        emb_size=flags.emb_size,
        save_dir=flags.save_dir,
        device=flags.device,
        lr=flags.learning_rate,
        gpu_mem_frac=flags.gpu_mem_frac,
        num_steps=flags.num_steps,
        summary_freq=flags.summary_freq,
        inference_freq=flags.inference_freq,
        batch_size=flags.batch_size,
        one_hot_embedding = flags.one_hot_embedding,
        char_embedding_size=flags.char_embedding_size,
        train_file=flags.train_file,
        val_file=flags.val_file,
        infer_file=flags.infer_file,
        allow_soft_placement=flags.allow_soft_placement,
        cpu_threads=flags.cpu_threads,
        overwrite_saves=flags.overwrite_saves,
        input_dropout=flags.input_dropout,
        emb_noise=flags.emb_noise,
        conv_hidden_size=flags.conv_hidden_size,
        kernel_size=flags.kernel_size,
        reverse_decoding=flags.reverse_decoding,
        buffer_size=flags.buffer_size,
        lr_decay=flags.lr_decay,
        lr_decay_frequency=flags.lr_decay_frequency,
        lr_decay_factor=flags.lr_decay_factor,
        num_buckets=flags.num_buckets,
        min_bucket_length=flags.min_bucket_length,
        max_bucket_length=flags.max_bucket_length,
        num_features=flags.num_features,
        rand_input_swap=flags.rand_input_swap,
        infer_input=flags.infer_input,
        emb_activation=flags.emb_activation,
        div_loss_scale=flags.div_loss_scale,
        div_loss_rate=flags.div_loss_rate,
    )
    hparams.add_hparam("encode_vocabulary_file", os.path.join(DEFAULT_DATA_DIR, "indices_char.npy"))
    hparams.add_hparam("decode_vocabulary_file", os.path.join(DEFAULT_DATA_DIR, "indices_char.npy"))
    hparams_file_name = flags.hparams_file_name
    if hparams_file_name is None:
        hparams_file_name = os.path.join(hparams.save_dir, 'hparams.json')
    if flags.hparams_from_file:
        hparams.cell_size = list()
        hparams = hparams.parse_json(json.load(open(hparams_file_name)))
        hparams.set_hparam("encode_vocabulary_file", os.path.join(DEFAULT_DATA_DIR, "indices_char.npy"))
        hparams.set_hparam("decode_vocabulary_file", os.path.join(DEFAULT_DATA_DIR, "indices_char.npy"))
    return hparams
