# Continuous and Data-Driven Descriptors (CDDD)

Implementation of the Paper "Learning Continuous and Data-Driven Molecular
Descriptors by Translating Equivalent Chemical Representations" by Robin Winter, Floriane Montanari, Frank Noe and Djork-Arne Clevert.


## Installing

### Prerequisites
```
python 3
tensorflow 1.10
numpy
rdkit
scikit-learn
```
### Conda
Create a new enviorment:
```
conda env create –f environment.yml
source activate cddd
```
Download and install the cddd package:
```
git clone git@by-gitlab.de.bayer.cnb:GGWAQ/cddd.git
cd cddd
pip install cddd
```

## Getting Started

Run the script run_cddd.py to extract molecular descripotrs of your provided SMILES:
```
run_cddd.py --input smiles.smi --output descriptors.csv  --smiles_header smiles
```
### Input
Supported input: 
  * .csv-file with one SMILES per row
  * .smi-file with one SMILES per row

For .csv: Specify the header of the SMILES column with the flag --smiles_header (default: smiles)


