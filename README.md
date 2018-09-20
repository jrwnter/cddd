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

Specify the header of the SMILES column with the flag --smiles_header (default: smiles)


