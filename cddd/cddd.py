from cddd.hyperparameters import add_arguments, create_hparams
from cddd.inference import sequence2embedding
from cddd.model_helper import build_models
import argparse

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