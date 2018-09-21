import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from cddd.inference import InferenceModel
from cddd.preprocessing import preprocess_smiles

FLAGS=None

def add_arguments(parser):
    parser.add_argument('-i', '--input', help='input .txt file with one SMILES per row.', type=str)
    parser.add_argument('-o', '--output', help='output .csv file with a descriptor for each SMILES per row.', type=str)
    parser.add_argument('--smiles_header', help='if .csv, specify the name of the SMILES column header here.', default="smiles", type=str)
    parser.add_argument('--preprocess', dest='preprocess', action='store_true')
    parser.add_argument('--no-preprocess', dest='preprocess', action='store_false')
    parser.set_defaults(preprocess=True)
    parser.add_argument('--model_path', default="default", type=str)
    parser.add_argument('--gpu', default=True, type=bool)
    parser.add_argument('--device', default="0", type=str)
    parser.add_argument('--batch_size', default=512, type=int)
    
def read_input():
    file = FLAGS.input
    if file.endswith('.csv'):
        df = pd.read_csv(file)
    if file.endswith('.smi'):
        df = pd.read_table(file, header=None).rename({0:FLAGS.smiles_header, 1:"ID"}, axis=1)
    return df
    
    
def main(unused_argv):
    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.device)
    if FLAGS.model_path == "default":
        model_path = None
    else:
        model_path = FLAGS.model_path
        
    df = read_input()
    if FLAGS.preprocess:
        print("start preprocessing SMILES...")
        df["new_smiles"] = df[FLAGS.smiles_header].map(preprocess_smiles)
        sml_list = df[~df.new_smiles.isna()].new_smiles.tolist()
        print("finished preprocessing SMILES!")
    else:
        sml_list = df[FLAGS.smiles_header].tolist()
    print("start calculating descriptors...")
    infer_model = InferenceModel(model_path=model_path, use_gpu=FLAGS.gpu, batch_size=FLAGS.batch_size)
    descriptors = infer_model.sml_to_emb(sml_list)
    print("finished calculating descriptors! %d out of %d input SMILES could be interpreted" %(len(sml_list), len(df)))
    
    if FLAGS.preprocess:
        df = df.join(pd.DataFrame(descriptors, index=df[~df.new_smiles.isna()].index, columns=["cddd_" + str(i+1) for i in range(512)]))
    else:
        df = df.join(pd.DataFrame(descriptors, index=df.index, columns=["cddd_" + str(i+1) for i in range(512)]))
    print("writing descriptors to file...")
    df.to_csv(FLAGS.output)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)