import argparse
import numpy as np
from cddd.input_pipeline import InputPipelineInferEncode, InputPipelineInferDecode
from cddd.hyperparameters import add_arguments, create_hparams
from cddd.model_helper import build_models
def sequence2embedding(model, hparams, seq_list):
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
        if (len(seq_list) == 1 ) & isinstance(seq_list, str):
            return seq_list[0]
    return seq_list

def calculate_descriptor(sml_list, model_path, batch_size=256, gpu_mem_frac=0.1):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    flags, unparsed = parser.parse_known_args()
    flags.hparams_from_file = True
    flags.save_dir = model_path
    hparams = create_hparams(flags)
    hparams.set_hparam("save_dir", model_path)
    hparams.set_hparam("batch_size", batch_size)
    hparams.set_hparam("gpu_mem_frac", gpu_mem_frac)
    encode_model = build_models(hparams, modes="ENCODE")
    embedding = sequence2embedding(encode_model, hparams, sml_list)
    return embedding

class InferenceModel():
    def __init__(self, model_path, use_gpu=True, batch_size=256, gpu_mem_frac=0.1, beam_width=10, num_top=1):
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
        self.encode_model, self.decode_model = build_models(self.hparams, modes=["ENCODE", "DECODE"])
        
    def sml_to_emb(self, smls):
        if isinstance(smls, str):
            smls = [smls]
        if self.use_gpu:
            emb = sequence2embedding(self.encode_model, self.hparams, smls)
        else:
            with tf.device("/cpu:0"):
                emb = sequence2embedding(self.encode_model, self.hparams, smls)
        return emb
    
    def emb_to_sml(self, embedding):
        if embedding.ndim == 1:
            embedding = np.expand_dims(embedding, 0) 
        if self.use_gpu:
            smls = embedding2sequence(self.decode_model, self.hparams, embedding, self.num_top)
        else:
            with tf.device("/cpu:0"):
                smls = embedding2sequence(self.decode_model, self.hparams, embedding, self.num_top)
        if len(smls) == 1:
            smls = smls[0]
        if len(smls) == 1:
            smls = smls[0]
        return smls 