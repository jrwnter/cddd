from collections import namedtuple
import tensorflow as tf
from .models import *
from .input_pipeline import *


def build_models(hparams, only_infer=False):
    if hparams.model == "GRUSeq2Seq":
        model = GRUSeq2Seq
    elif hparams.model == "LSTMSeq2Seq":
        model = LSTMSeq2Seq
    elif hparams.model == "Conv2GRUSeq2Seq":
        model = Conv2GRUSeq2Seq
    elif hparams.model == "NoisyGRUSeq2Seq":
        model = NoisyGRUSeq2Seq
    elif hparams.model == "GRUSeq2SeqWithFeatures":
        model = GRUSeq2SeqWithFeatures
    elif hparams.model == "NoisyGRUSeq2SeqWithFeatures":
        model = NoisyGRUSeq2SeqWithFeatures
    else:
        raise ValueError("Model %s not known!" %(hparams.model))
        
    if hparams.input_pipeline == "SmlToCanSml":
        input_pipeline = InputPipelineSmlToCanSml
    elif hparams.input_pipeline == "CanSmlToCanSml":
        input_pipeline = InputPipelineCanSmlToCanSml
    elif hparams.input_pipeline == "SmlToCanSmlWithFeatures":
        input_pipeline = InputPipelineSmlToCanSmlWithFeatures
    elif hparams.input_pipeline == "CanSmlToCanSmlWithFeatures":
        input_pipeline = InputPipelineCanSmlToCanSmlWithFeatures
    elif hparams.input_pipeline == "InchiToCanSml":
        input_pipeline = InputPipelineInchiToCanSml
    elif hparams.input_pipeline == "InchiToCanSmlWithFeatures":
        input_pipeline = InputPipelineInchiToCanSmlWithFeatures
    elif hparams.input_pipeline == "CanSmlToInchi":
        input_pipeline = InputPipelineCanSmlToInchi
    else:
        raise ValueError("Input_pipeline %s not known" %(hparams.input_pipeline))
        
    if only_infer:
        infer_model = create_infer_model(model, hparams)
        return infer_model
    else:   
        train_model = create_train_model(model, input_pipeline, hparams)
        eval_model = create_eval_model(model, input_pipeline, hparams)
        infer_model = create_infer_model(model, hparams)
        return train_model, eval_model, infer_model


class TrainModel(namedtuple("TrainModel", ("graph", "model", "sess"))):
    pass

class EvalModel(namedtuple("EvalModel", ("graph", "model", "sess"))):
    pass

class InferModel(namedtuple("InferModel", ("graph", "model", "sess"))):
    pass

def create_train_model(model_creator, input_pipeline_creator, hparams):
    sess_config = tf.ConfigProto(allow_soft_placement=hparams.allow_soft_placement,
                                 gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=hparams.gpu_mem_frac),
                                 inter_op_parallelism_threads=hparams.cpu_threads,
                                 intra_op_parallelism_threads=hparams.cpu_threads)
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        input_pipeline = input_pipeline_creator("TRAIN", hparams)
        input_pipeline.make_dataset_and_iterator()
        model = model_creator(mode="TRAIN",
                              iterator=input_pipeline.iterator,
                              hparams=hparams
                             )
        model.build_graph()
    sess = tf.Session(graph=graph, config=sess_config)
    return TrainModel(graph=graph, model=model, sess=sess)

def create_eval_model(model_creator, input_pipeline_creator, hparams):
    sess_config = tf.ConfigProto(allow_soft_placement=hparams.allow_soft_placement,
                                 gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=hparams.gpu_mem_frac),
                                 inter_op_parallelism_threads=hparams.cpu_threads,
                                 intra_op_parallelism_threads=hparams.cpu_threads)
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        input_pipeline = input_pipeline_creator("EVAL", hparams)
        input_pipeline.make_dataset_and_iterator()
        model = model_creator(mode="EVAL",
                              iterator=input_pipeline.iterator,
                              hparams=hparams
                             )
        model.build_graph()
    sess = tf.Session(graph=graph, config=sess_config)
    return EvalModel(graph=graph, model=model, sess=sess)

def create_infer_model(model_creator, hparams):
    sess_config = tf.ConfigProto(allow_soft_placement=hparams.allow_soft_placement,
                                 gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=hparams.gpu_mem_frac),
                                 inter_op_parallelism_threads=hparams.cpu_threads,
                                 intra_op_parallelism_threads=hparams.cpu_threads)
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        model = model_creator(mode="INFER",
                              iterator=None,
                              hparams=hparams
                             )
        model.build_graph()
    sess = tf.Session(graph=graph, config=sess_config)
    return InferModel(graph=graph, model=model, sess=sess)

