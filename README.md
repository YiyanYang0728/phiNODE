# phiNODE: Advancing the Discovery of Phage-Host Interactions and Disease Classification from Metagenomic Profiles Using Deep Learning

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Dependencies](#dependencies)
- [License](#license)
- [Citation](#citation)

## Installation
To install phiNODE, follow these steps:

1. Install the required dependencies:
```
conda env create --file environment.yml
# or if you have mamba installed
# mamba env create --file environment.yml
```

2. Clone the repository:
```
git clone git@github.com:YiyanYang0728/phiNODE.git
```

3. Change into the directory:
```
cd phiNODE
```

## Usage
### Step 1. Prepare data
All parameters are provided in a config file:
"""
data:
  train_X: data/Phage_train.tsv
  train_Y: data/Bact_arc_train.tsv
  val_X: data/Phage_val.tsv
  val_Y: data/Bact_arc_val.tsv
model:
  batch_size: 512 # Batch size
  patience: 20 # Number of epochs with no improvement after which training will be stopped
  num_epochs: 50000 # Maximal epochs for training
  n_trials: 10 # Number of trials for choosing the optimal combinations of parameters, default=10
  path: results/phiNODE_best_model.pth
  para_path: results/phiNODE_best_params.yaml

"""
```
python split_data.py raw_data/xxx.tsv raw_data/yyy.tsv # [TBA]
```

### Step 2. Train model
```
# copy and modify config_train.yaml
cp config/config_train.yaml my_config_train.yaml 
# run training script
python model/train.py -c my_config_train.yaml
```
The best model and parameters are in `results/phiNODE_best_model.pth` and `results/phiNODE_best_params.yaml`.
The historical models and training log file are in `checkpoints/`.

### Step 3. Test and validate the model
```
# copy and paste the best parameters to config file: config_test.yaml
awk '{print "  "$0}' results/phiNODE_best_params.yaml | cat config/config_test.yaml - > my_config_test.yaml
# run testing script
python model/test.py -c my_config_test.yaml

# Similar steps for validation and training data, uncomment to run the command lines
# awk '{print "  "$0}' results/phiNODE_best_params.yaml | cat config/config_test.val.yaml - > my_config_test.val.yaml
# python model/test.py -c my_config_test.val.yaml
# awk '{print "  "$0}' results/phiNODE_best_params.yaml | cat config/config_test.train.yaml - > my_config_test.train.yaml
# python model/test.py -c my_config_test.train.yaml
```
```
# To summarize results
scripts/summarize.sh results/phiNODE_predict_test &> results/phiNODE_predict_test.ft.metrics
# Similar steps for validation and training data, uncomment to run the command lines
# scripts/summarize.sh results/phiNODE_predict_val &> results/phiNODE_predict_val.ft.metrics
# scripts/summarize.sh results/phiNODE_predict_train &> results/phiNODE_predict_train.ft.metrics
```

### Step 4. Predict interactions
```
# To infer interactions from training samples, make sure you have prepared results/phiNODE_predict_train_pred.tsv by running: python model/test.py -c config_test.train.yaml
# get the host sensitivity values for each phage feature
awk '{print NR-1}' data/Phage_feature_names.txt | parallel scripts/infer_interactions.1.sh {}

# summarize and organize the results
N_train=`cat data/train_samples.txt|wc -l`
phage_ct=`cat data/Phage_feature_names.txt |wc -l`
prok_ct=`cat data/Bact_arc_feature_names.txt |wc -l`
res_dir=results
scripts/infer_interactions.2.sh ${N_train} ${phage_ct} ${prok_ct} ${res_dir}
```
The intermediate files are in `perturb/`.
The predicted interactions are in `results/predicted_interactions.tsv`.

### Step 5. Extract latent representations
```
outdim=`cat data/Bact_arc_feature_names.txt |wc -l`
awk '{print "  "$0}' results/phiNODE_best_params.yaml | cat config/config_repr.train.yaml - > my_config_repr.train.yaml
sed -i "s|#OUTDIM#|${outdim}|g" my_config_repr.train.yaml
python model/extract_repr.py -c my_config_repr.train.yaml
# Similar steps for validation and training data, uncomment to run the command lines
awk '{print "  "$0}' results/phiNODE_best_params.yaml | cat config/config_repr.val.yaml - > my_config_repr.val.yaml
sed -i "s|#OUTDIM#|${outdim}|g" my_config_repr.val.yaml
python model/extract_repr.py -c my_config_repr.val.yaml

awk '{print "  "$0}' results/phiNODE_best_params.yaml | cat config/config_repr.test.yaml - > my_config_repr.test.yaml
sed -i "s|#OUTDIM#|${outdim}|g" my_config_repr.test.yaml
python model/extract_repr.py -c my_config_repr.test.yaml

# if you need the latent representations for all samples
cat results/phiNODE_train_repr1.tsv results/phiNODE_val_repr1.tsv results/phiNODE_test_repr1.tsv > results/merged_repr1.tsv
cat results/phiNODE_train_repr2.tsv results/phiNODE_val_repr2.tsv results/phiNODE_test_repr2.tsv > results/merged_repr2.tsv
```

## Dependencies

phiNODE requires the following Python libraries:
- Pytorch
- torchdiffeq
- datamash

## License

phiNODE is released under the [MIT License](./LICENSE).

## Citation

If you use phiNODE in your research, please cite it as follows:
TBA
