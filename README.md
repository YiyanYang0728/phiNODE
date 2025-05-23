# phiNODE: Advancing the Discovery of Phage-Host Interactions and Disease Classification from Metagenomic Profiles Using Deep Learning

## Overview

**phiNODE** is a deep-learning framework designed to predict phage-host interactions and classify sample status using latent representations from trained model. 

## Table of Contents

* [Installation](#installation)
* [Usage](#usage)

  * [1. Data Preparation](#1-data-preparation)
  * [2. Model Training](#2-model-training)
  * [3. Evaluation](#3-evaluation)
  * [4. Interaction Inference](#4-interaction-inference)
  * [5. Latent Representation Extraction (Optional)](#5-latent-representation-extraction-optional)
* [Dependencies](#dependencies)
* [License](#license)
* [Citation](#citation)

## Installation

**Create a Conda Environment**

   ```bash
   conda env create --file environment.yml
   # or, with mamba:
   mamba env create --file environment.yml
   ```

   > **Attetion:** The default environment is configured for NVIDIA A- and V-series GPUs. To run on CPU-only systems, remove CUDA-related packages from `environment.yml` before creating the environment.

**Clone the Repository**

   ```bash
   git clone git@github.com:YiyanYang0728/phiNODE.git
   cd phiNODE
   ```

## Usage

### 1. Data Preparation

Prepare three inputs for `scripts/split_data.py`: the prokaryotic abundance profile, the phage abundance profile, and an output directory.

```bash
python scripts/split_data.py \
    raw_data/Bact_arc_profile.tsv \
    raw_data/Phage_profile.tsv \
    data/
```

### 2. Model Training

Configure training parameters and file paths in a YAML file.

```bash
# Copy the template and customize
cp config/config_train.yaml my_config_train.yaml

# Run training
python model/train.py -c my_config_train.yaml
```

* **Outputs:**

  * Best model: `results/phiNODE_best_model.pth`
  * Best parameters: `results/phiNODE_best_params.yaml`
  * Checkpoints and logs: `checkpoints/`

### 3. Evaluation

Generate Test Predictions

```bash
# Prepare test config by merging best parameters
awk '{print "  "$0}' results/phiNODE_best_params.yaml \
    | cat config/config_test.yaml - \
    > my_config_test.yaml

# Run testing
python model/test.py -c my_config_test.yaml

# Summarize metrics
scripts/summarize.sh \
    results/phiNODE_predict_test \
    &> results/phiNODE_predict_test.ft.metrics
```

> **Tip:** Repeat the above steps for validation and training sets by using `config_test.val.yaml` / `config_test.train.yaml` and corresponding prediction directories.
```bash
awk '{print "  "$0}' results/phiNODE_best_params.yaml \
    | cat config/config_val.yaml - \
    > my_config_val.yaml
python model/test.py -c my_config_val.yaml
scripts/summarize.sh \
    results/phiNODE_predict_val \
    &> results/phiNODE_predict_val.ft.metrics

awk '{print "  "$0}' results/phiNODE_best_params.yaml \
    | cat config/config_train.yaml - \
    > my_config_train.yaml
python model/test.py -c my_config_train.yaml
scripts/summarize.sh \
    results/phiNODE_predict_train \
    &> results/phiNODE_predict_train.ft.metrics
```

### 4. Interaction Inference

Infer host sensitivity for each phage feature and compile interaction tables.

```bash
# Ensure predictions exist: results/phiNODE_predict_train_pred.tsv
awk '{print NR-1}' data/Phage_feature_names.txt \
    | parallel scripts/infer_interactions.1.sh {}

# Summarize interactions
N_train=$(wc -l < data/train_samples.txt)
phage_ct=$(wc -l < data/Phage_feature_names.txt)
prok_ct=$(wc -l < data/Bact_arc_feature_names.txt)
res_dir=results
scripts/infer_interactions.2.sh $N_train $phage_ct $prok_ct $res_dir
```

* **Output:**

  * Intermediate files: `perturb/`
  * Final interactions: `results/predicted_interactions.tsv`

### 5. Latent Representation Extraction (Optional)

Extract sample embeddings from the trained model.

```bash
# Determine output dimension
outdim=$(wc -l < data/Bact_arc_feature_names.txt)

# Prepare representation config
awk '{print "  "$0}' results/phiNODE_best_params.yaml \
    | cat config/config_repr.train.yaml - \
    > my_config_repr.train.yaml
sed -i "s|#OUTDIM#|$outdim|g" my_config_repr.train.yaml

# Extract representations for training data
python model/extract_repr.py -c my_config_repr.train.yaml
```

> **Tip:** Apply the same process for validation and test sets by using `config_repr.val.yaml` and `config_repr.test.yaml`. 

Merge representations.

```bash
cat results/phiNODE_train_repr1.tsv \
    results/phiNODE_val_repr1.tsv \
    results/phiNODE_test_repr1.tsv \
    > results/merged_repr1.tsv

cat results/phiNODE_train_repr2.tsv \
    results/phiNODE_val_repr2.tsv \
    results/phiNODE_test_repr2.tsv \
    > results/merged_repr2.tsv
```

## Dependencies

* [PyTorch](https://github.com/pytorch/pytorch)
* [torchdiffeq](https://github.com/rtqichen/torchdiffeq)

## License

This project is released under the [MIT License](./LICENSE).

## Citation

Please cite phiNODE as follows:

> TBA
