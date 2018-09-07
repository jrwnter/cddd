from collections import namedtuple
import tensorflow as tf
from cddd import models
from cddd import input_pipeline


def build_models(hparams, modes = ["TRAIN", "EVAL", "ENCODE"]):
    model = getattr(models, hparams.model)
    input_pipe = getattr(input_pipeline, hparams.input_pipeline)
    model_list = []
    if isinstance(modes, list):
        for mode in modes:
            model_list.append(create_model(mode, model, input_pipe, hparams))
        return tuple(model_list)
    else:
        return create_model(modes, model, input_pipe, hparams)

Model = namedtuple("Model", ("graph", "model", "sess"))

def create_model(mode, model_creator, input_pipeline_creator, hparams):
    sess_config = tf.ConfigProto(allow_soft_placement=hparams.allow_soft_placement,
                                 gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=hparams.gpu_mem_frac),
                                 inter_op_parallelism_threads=hparams.cpu_threads,
                                 intra_op_parallelism_threads=hparams.cpu_threads)
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        if mode in ["TRAIN", "EVAL"]:
            input_pipe = input_pipeline_creator(mode, hparams)
            input_pipe.make_dataset_and_iterator()
            iterator = input_pipe.iterator
        else:
            iterator = None
        model = model_creator(mode=mode,
                              iterator=iterator,
                              hparams=hparams
                             )
        model.build_graph()
    sess = tf.Session(graph=graph, config=sess_config)
    return Model(graph=graph, model=model, sess=sess)
