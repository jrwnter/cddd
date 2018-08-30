import csv
import os
import numpy as np
import tensorflow as tf


def idx_to_char(seq, voc, voc_inverse):
    return ''.join([voc[idx] for idx in seq if ((idx != voc_inverse['</s>']) & (idx != voc_inverse['<s>']))])
    #return ''.join([voc[idx] for idx in seq ])
def write_to_logfile(model, step, hparams):
    voc_encode = np.load(hparams.encode_vocabulary_file).item()
    voc_decode = np.load(hparams.decode_vocabulary_file).item()
    voc_inverse_encode = {v: k for k, v in voc_encode.items()}
    voc_inverse_decode = {v: k for k, v in voc_decode.items()}
    fields = [step]
    header = ["step"] + list(model.model.measures_to_log.keys())
    if step == 0:
        with open(os.path.join(hparams.save_dir, model.model.mode + ".csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
    measures = []
    while True:
        try:
            measures.append(model.model.eval(model.sess))
        except tf.errors.OutOfRangeError:
            break
            
    fields.extend(np.mean(measures, axis=0).tolist())
    
    # map smiles sequence back to characters
    #fields[header.index("input_sequence")] = idx_to_char(fields[header.index("input_sequence")][0], voc_encode, voc_inverse_encode)
    #fields[header.index("target_sequence")] = idx_to_char(fields[header.index("target_sequence")][0], voc_decode, voc_inverse_decode)
    #fields[header.index("predicted_sequence")] = idx_to_char(fields[header.index("predicted_sequence")][0], voc_decode, voc_inverse_decode)
    
    with open(os.path.join(hparams.save_dir, model.model.mode + ".csv"), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

        
