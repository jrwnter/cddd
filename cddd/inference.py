"""Helper functions to run inference on trained models"""
import argparse
import os
import numpy as np
import tensorflow as tf
from cddd.input_pipeline import InputPipelineInferEncode, InputPipelineInferDecode
from cddd.hyperparameters import add_arguments, create_hparams
from cddd.model_helper import build_models

def sequence2embedding(model, hparams, seq_list):
    """Helper Function to run a forwards path up to the bottneck layer (ENCODER).
    Encodes a list of sequnces into the molecular descriptor.

    Args:
        model: The translation model instance to use.
        hparams: Hyperparameter object.
        seq_list: list of sequnces that should be encoded.
    Returns:
        Embedding of the input sequnces as numpy array.
    """
    emb_list = []
    with model.graph.as_default():
        input_pipeline = InputPipelineInferEncode(seq_list, hparams)
        input_pipeline.initilize()
        model.model.restore(model.sess)
        while 1:
            try:
                input_seq, input_len = input_pipeline.get_next()
                emb = model.model.seq2emb(model.sess, input_seq, input_len)
                emb_list.append(emb)
            except StopIteration:
                break
        embedding_array = np.concatenate(emb_list)
    return embedding_array

def embedding2sequence(model, hparams, embedding, num_top=1):
    """Helper Function to run a forwards path from thebottneck layer to
    output (DECODER).

    Args:
        model: The translation model instance to use.
        hparams: Hyperparameter object.
        embedding: Array with samples x num_features
    Returns:
        List of sequences decoded from the input embedding (descriptor).
    """
    seq_list = []
    with model.graph.as_default():
        input_pipeline = InputPipelineInferDecode(embedding, hparams)
        input_pipeline.initilize()
        model.model.restore(model.sess)
        while 1:
            try:
                emb = input_pipeline.get_next()
                seq = model.model.emb2seq(model.sess, emb, num_top)
                if num_top == 1:
                    seq = [s[0] for s in seq]
                seq_list.extend(seq)
            except StopIteration:
                break
        if (len(seq_list) == 1) & isinstance(seq_list, str):
            return seq_list[0]
    return seq_list

class InferenceModel(object):
    """Class that handles the inference of a trained model."""
    def __init__(self, model_path=None, use_gpu=True, batch_size=256,
                 gpu_mem_frac=0.1, beam_width=10, num_top=1, emb_activation=None):
        """Constructor for the inference model.

        Args:
            model_path: Path to the model save file.
            use_gpu: Flag for GPU usage.
            batch_size: Number of samples to process per step.
            gpu_mem_frac: If GPU is used, what memory fraction should be used?
            beam_width:  Width of the the window used for the beam search decoder.
            num_top: Number of most probable sequnces as output of the beam search decoder.
            emb_activation: Activation function used in the bottleneck layer.
        Returns:
            None
        """
        if model_path is None:
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'default_model'))
        self.num_top = num_top
        self.use_gpu = use_gpu
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        flags = parser.parse_args([])
        flags.hparams_from_file = True
        flags.save_dir = model_path
        self.hparams = create_hparams(flags)
        self.hparams.set_hparam("save_dir", model_path)
        self.hparams.set_hparam("batch_size", batch_size)
        self.hparams.set_hparam("gpu_mem_frac", gpu_mem_frac)
        self.hparams.add_hparam("beam_width", beam_width)
        if emb_activation is not None:
            self.hparams.set_hparam("emb_activation", emb_activation)
        self.encode_model, self.decode_model = build_models(self.hparams,
                                                            modes=["ENCODE", "DECODE"])

    def seq_to_emb(self, seq):
        """Helper function to calculate the embedding (molecular descriptor) for input sequnce(s)

        Args:
            seq: Single sequnces or list of sequnces to encode.
        Returns:
            Embedding of the input sequnce(s).
        """
        if isinstance(seq, str):
            seq = [seq]
        if self.use_gpu:
            emb = sequence2embedding(self.encode_model, self.hparams, seq)
        else:
            with tf.device("/cpu:0"):
                emb = sequence2embedding(self.encode_model, self.hparams, seq)
        return emb

    def emb_to_seq(self, embedding):
        """Helper function to calculate the sequnce(s) for one or multiple (concatinated)
        embedding.

        Args:
            embedding: array with n_samples x num_features.
        Returns:
            sequnce(s).
        """
        if embedding.ndim == 1:
            embedding = np.expand_dims(embedding, 0)
        if self.use_gpu:
            seq = embedding2sequence(self.decode_model, self.hparams, embedding, self.num_top)
        else:
            with tf.device("/cpu:0"):
                seq = embedding2sequence(self.decode_model, self.hparams, embedding, self.num_top)
        if len(seq) == 1:
            seq = seq[0]
        if len(seq) == 1:
            seq = seq[0]
        return seq
    