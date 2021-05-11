# Continuous and Data-Driven Descriptors (CDDD)

Implementation of the Paper "Learning Continuous and Data-Driven Molecular
Descriptors by Translating Equivalent Chemical Representations" by Robin Winter, Floriane Montanari, Frank Noe and Djork-Arne Clevert.<sup>1</sup>

<img src="example/model.png" width="75%" height="75%">

## Installing

### Prerequisites
```
python 3.6
tensorflow 1.10
numpy<=1.14.5
pandas<=1.0.3
rdkit
scikit-learn
```
### Conda
Create a new enviorment with above mentioned prerequisites and:
```bash
pip install cddd  # with GPU support
```
or 
```bash
pip install cddd-cpu  # without GPU support
```

Or simply utilize the repspective .yml file:
```bash
conda env create -f environment_gpu.yml  # with GPU support
```
or 
```bash
conda env create -f environment_cpu.yml  # without GPU support
```

### Downloading Pretrained Model
A pretrained model as described in ref. 1 is available on Google Drive. It is automatically downloaded to the default
user data dir (as defined by appdirs). It can also wbe download and unzip by execuiting the bash script "download_default_model.sh":
```bash
./download_default_model.sh
```
The default_model.zip file can also be downloaded manualy under https://drive.google.com/open?id=1oyknOulq_j0w9kzOKKIHdTLo5HphT99h
### Testing
Extract molecular descriptors from two QSAR datasets (ref. 2,3) and evaluate the perfromance of a SVM trained on these descriptors.
```bash
cd example
python run_qsar_test.py
```
or with gpu support on e.g. device 0:
```
python run_qsar_test.py --use_gpu --device 0
```
The accuracy on the Ames dataset should be arround 0.814 +/- 0.006.

The r2 on the Lipophilicity dataset should be arround 0.731 +/- 0.029.

## Getting Started
### Extracting Molecular Descripotrs
Run the script run_cddd.py to extract molecular descripotrs of your provided SMILES:
```bash
cddd --input smiles.smi --output descriptors.csv  --smiles_header smiles
```
Supported input: 
  * .csv-file with one SMILES per row
  * .smi-file with one SMILES per row

For .csv: Specify the header of the SMILES column with the flag --smiles_header (default: smiles)

### Inference Module
The pretrained model can also be imported and used directly in python via the inference class:
```python
import pandas as pd
from cddd.inference import InferenceModel
from cddd.preprocessing import preprocess_smiles
```
Load and preprocess data:
```python
ames_df = pd.read_csv("example/ames.csv", index_col=0)
ames_df["smiles_preprocessed"] = ames_df.smiles.map(preprocess_smiles)
ames_df = ames_df.dropna()
smiles_list = ames_df["smiles_preprocessed"].tolist()
```
Create a instance of the inference class:
```python
inference_model = InferenceModel()
```
Encode all SMILES into the continuous embedding (molecular descriptor):
```python
smiles_embedding = inference_model.seq_to_emb(smiles_list)
```
The infernce model instance can also be used to decode a molecule embedding back to a interpretable SMILES string:
```python
decoded_smiles_list = inference_model.emb_to_seq(smiles_embedding)
```
### References
[1] R. Winter, F. Montanari, F. Noe and D. Clevert, Chem. Sci, 2019, https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc04175j#!divAbstract

[2] K. Hansen, S. Mika, T. Schroeter, A. Sutter, A. Ter Laak, T. Steger-Hartmann, N. Heinrich and K.-R. MuÌ´Lller, J. Chem. Inf. Model., 2009, 49, 2077–2081.

[3] Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse, A. S. Pappu, K. Leswing and V. Pande, Chemical Science, 2018, 9, 513–530.

