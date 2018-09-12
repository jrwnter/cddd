import os
import sys
import argparse
import warnings
import numpy as np
import tensorflow as tf
from cddd.inference import InferenceModel
from cddd.preprocessing import preprocess_smiles

FLAGS=None

def add_arguments(parser):
    parser.add_argument('-i', '--input', help='input .txt file with one SMILES per row.', type=str)
    parser.add_argument('-o', '--output', help='output .txt file with a descriptor for each SMILES per row, delimeted by tab.', type=str)
    parser.add_argument('--preprocess', default=True, type=bool)
    parser.add_argument('--model_path', default="default", type=str)
    parser.add_argument('--gpu', default=True, type=bool)
    parser.add_argument('--device', default="0", type=str)
    parser.add_argument('--batch_size', default=512, type=int)
    
def preprocess(sml_list):
    sml_list2 = []
    fail_ids = []
    for i, sml in enumerate(sml_list):
        new_sml = preprocess_smiles(sml)
        if isinstance(new_sml, str):
            sml_list2.append(new_sml)
        else:
            fail_ids.append(i)
    num_fails = len(fail_ids)
    if  num_fails > 0:
        warnings.warn("Warning: The input file contains {} SMILES that could not be interpreted. Outputting nan....".format(num_fails))
    return sml_list2, fail_ids
    
def main(unused_argv):
    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.device)
    print("reading in SMILES...")
    with open(FLAGS.input) as f:
        sml_list = f.read().splitlines()
    if FLAGS.preprocess:
        print("start preprocessing SMILES...")
        sml_list, fail_ids = preprocess(sml_list)
    print("finished preprocessing SMILES!")
    if FLAGS.model_path == "default":
        model_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data', 'default_model'))
    else:
        model_path = FLAGS.model_path
    print("start calculating descriptors...")
    infer_model = InferenceModel(model_path=model_path, use_gpu=FLAGS.gpu, batch_size=FLAGS.batch_size)
    descriptors = infer_model.sml_to_emb(sml_list)
    print("finished calculating descriptors!")
    nan_arr = np.ones(descriptors.shape[1]) * np.float("nan")
    for i in fail_ids:
        descriptors = np.insert(descriptors, i, nan_arr, 0)
    print("writing descriptors to file...")
    np.savetxt(FLAGS.output, descriptors, delimiter="\t")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)