import argparse
import tensorflow as tf
import numpy as np
from rdkit import Chem
from sklearn.preprocessing import scale
from seq2seq.inference import simple_inference, simple_inference_decode
from seq2seq.smilesseq2seq import add_arguments, create_hparams
from seq2seq.model_helper import build_models, create_infer_model
from seq2seq.models import OptModel

def seq2seq_descriptor_gpu(sml_list, model_path, batch_size=256):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    flags, unparsed = parser.parse_known_args()
    flags.hparams_from_file = True
    flags.save_dir = model_path
    hparams = create_hparams(flags)
    hparams.set_hparam("save_dir", model_path)
    hparams.set_hparam("decode_vocabulary_file", '/gpfs01/home/ggwaq/smiles/seq2seq/indices_char.npy')
    hparams.set_hparam("encode_vocabulary_file", '/gpfs01/home/ggwaq/smiles/seq2seq/indices_char.npy')
    hparams.set_hparam("batch_size", batch_size)
    #hparams.set_hparam("device", -1)
    hparams.set_hparam("gpu_mem_frac", 0.1)
    hparams.set_hparam("rand_input_swap", False)
    infer_model = build_models(hparams, only_infer=True)
    return simple_inference(infer_model, hparams, sml_list)

def seq2seq_descriptor_cpu(sml_list, model_path, batch_size=256):
    with tf.device("/cpu:0"):
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        flags, unparsed = parser.parse_known_args()
        flags.hparams_from_file = True
        flags.save_dir = model_path
        hparams = create_hparams(flags)
        hparams.set_hparam("save_dir", model_path)
        hparams.set_hparam("decode_vocabulary_file", '/gpfs01/home/ggwaq/smiles/seq2seq/indices_char.npy')
        hparams.set_hparam("encode_vocabulary_file", '/gpfs01/home/ggwaq/smiles/seq2seq/indices_char.npy')
        hparams.set_hparam("batch_size", batch_size)
        #hparams.set_hparam("device", -1)
        hparams.set_hparam("gpu_mem_frac", 0.1)
        hparams.set_hparam("rand_input_swap", False)
        infer_model = build_models(hparams, only_infer=True)
        return simple_inference(infer_model, hparams, sml_list)
    
    
class InferModel():
    def __init__(self, gpu=True, batch_size=256, num_top=1, model_path= "/gpfs01/home/ggwaq/smiles/seq2seq/saves2/features_noisy_gru_seq2seq_run7", decoder_voc='/gpfs01/home/ggwaq/smiles/seq2seq/indices_char.npy', encoder_voc='/gpfs01/home/ggwaq/smiles/seq2seq/indices_char.npy'):
        self.model_path = model_path
        self.batch_size = batch_size
        self.decoder_voc = decoder_voc
        self.encoder_voc = encoder_voc
        self.num_top = num_top
        
    def build(self):
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        flags, unparsed = parser.parse_known_args()
        flags.hparams_from_file = True
        flags.save_dir = self.model_path
        self.hparams = create_hparams(flags)
        self.hparams.set_hparam("save_dir", self.model_path)
        self.hparams.set_hparam("decode_vocabulary_file", self.decoder_voc)
        self.hparams.set_hparam("encode_vocabulary_file", self.encoder_voc)
        self.hparams.set_hparam("batch_size", self.batch_size)
        #hparams.set_hparam("device", -1)
        self.hparams.set_hparam("gpu_mem_frac", 0.1)
        self.hparams.set_hparam("rand_input_swap", False)
        self.encoder_model = build_models(self.hparams, only_infer=True)
        self.hparams.add_hparam("beam_width", 10)
        self.decoder_model = create_infer_model(OptModel, self.hparams)
    
    def sml_to_emb(self, smls):
        if isinstance(smls, str):
            smls = [smls]
        return simple_inference(self.encoder_model, self.hparams, smls)
    
    def emb_to_sml(self, emb):
        if emb.ndim == 1:
            emb = np.expand_dims(emb, 0)
        smls = simple_inference_decode(self.decoder_model, self.hparams, emb, self.num_top)
        if len(smls) == 1:
            smls = smls[0]
        if len(smls) == 1:
            smls = smls[0]
        return smls