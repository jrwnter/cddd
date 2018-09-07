import argparse
import numpy as np
from cddd.input_pipeline import InputPipelineInferEncode, InputPipelineInferDecode

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