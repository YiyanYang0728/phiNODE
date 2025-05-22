import numpy as np
import pandas as pd
import sys, os
from skbio.stats.composition import clr

"""
Split microbial profile tables into training, validation, and test sets.

This script divides two input profile tables (bacterial and phage) into 
train/val/test subsets according to a default ratio of 8:1:1. You can 
optionally filter out low-abundance features by setting a relative 
abundance threshold.

Usage:
    python split_data.py <bacterial_table> <phage_table> <output_dir> [--threshold THRESHOLD]
Example:
    python split_data.py data/bacteria.tsv data/phages.tsv results 1e-5
"""

def clr_data(df):
    dt = df.to_numpy()
    dt = dt + 1e-10
    dt_norm = dt / dt.sum(axis=1, keepdims=True) # re-normalize the data, along columns (species) to get the sum for each row (sample)
    dt_norm_1 = dt_norm * 1e6
    transformed_dt = clr(dt_norm_1 + 1) # Method 1 to normalization
    # transformed_dt = np.log2(dt_norm_1 + 1) # Method 2 to normalization, uncomment to use this method
    dt_norm1_df = pd.DataFrame(data=dt_norm, index=df.index, columns=df.columns)
    transformed_df = pd.DataFrame(data=transformed_dt, index=df.index, columns=df.columns)
    return dt_norm1_df, transformed_df

# row: species; column: samples
Bact_arc_df = pd.read_table(sys.argv[1], sep="\t", index_col=0)
Phage_df = pd.read_table(sys.argv[2], sep="\t", index_col=0)
outdir=sys.argv[3]
if len(sys.argv) == 5:
    ab_cutoff = float(sys.argv[4])
    print("Will filter dataset based on cutoff:", ab_cutoff)
else:
    ab_cutoff = None
os.makedirs(outdir, exist_ok=True)

for c in Bact_arc_df.columns:
    Bact_arc_df[c] = pd.to_numeric(Bact_arc_df[c])
for c in Phage_df.columns:
    Phage_df[c] = pd.to_numeric(Phage_df[c])

if ab_cutoff:
    Bact_arc_df[Bact_arc_df < ab_cutoff] = 0.
    Phage_df[Phage_df < ab_cutoff] = 0.

# remove features wilth all zeros (along col for each row)
Bact_arc_df = Bact_arc_df.loc[(Bact_arc_df.sum(axis=1) != 0),:]
Phage_df = Phage_df.loc[(Phage_df.sum(axis=1) != 0),:]

# remove samples with all PHAGE taxa are zeros (along row for each col)
Bact_arc_df = Bact_arc_df.loc[:,(Phage_df.sum(axis=0) != 0)]
Phage_df = Phage_df.loc[:,(Phage_df.sum(axis=0) != 0)]

# shuffle and split samples
np.random.seed(777)
samples = np.intersect1d(Bact_arc_df.columns.values, Phage_df.columns.values)
np.random.shuffle(samples)
num_samples = len(samples)

num_train_samples = int(len(samples)*0.8)
num_validation_samples = int(len(samples)*0.1)
num_test_samples = len(samples) - num_train_samples - num_validation_samples
print(num_train_samples, num_validation_samples, num_test_samples)

train_samples = samples[:num_train_samples]
val_samples = samples[num_train_samples:(num_train_samples+num_validation_samples)]
test_samples = samples[(num_train_samples+num_validation_samples):]

# row: samples; column: species
Bact_arc_train_df = Bact_arc_df[train_samples].T
Bact_arc_val_df = Bact_arc_df[val_samples].T
Bact_arc_test_df = Bact_arc_df[test_samples].T

Phage_train_df = Phage_df[train_samples].T
Phage_val_df = Phage_df[val_samples].T
Phage_test_df = Phage_df[test_samples].T

# Centered Log-Ratio transformed data
# now data row is sample, col is feature
# profile is composition  sum == 1; df is clr data.
Bact_arc_train_profile, Bact_arc_train_df = clr_data(Bact_arc_train_df)
Bact_arc_val_profile, Bact_arc_val_df = clr_data(Bact_arc_val_df)
Bact_arc_test_profile, Bact_arc_test_df = clr_data(Bact_arc_test_df)
Phage_train_profile, Phage_train_df = clr_data(Phage_train_df)
Phage_val_profile, Phage_val_df = clr_data(Phage_val_df)
Phage_test_profile, Phage_test_df = clr_data(Phage_test_df)

print(Bact_arc_train_df.shape, Bact_arc_val_df.shape, Bact_arc_test_df.shape, Phage_train_df.shape, Phage_val_df.shape, Phage_test_df.shape)

Bact_arc_train_df.to_csv(outdir+"/Bact_arc_train.tsv", sep="\t", header=False, index=False)
Bact_arc_val_df.to_csv(outdir+"/Bact_arc_val.tsv", sep="\t", header=False, index=False)
Bact_arc_test_df.to_csv(outdir+"/Bact_arc_test.tsv", sep="\t", header=False, index=False)
Phage_train_df.to_csv(outdir+"/Phage_train.tsv", sep="\t", header=False, index=False)
Phage_val_df.to_csv(outdir+"/Phage_val.tsv", sep="\t", header=False, index=False)
Phage_test_df.to_csv(outdir+"/Phage_test.tsv", sep="\t", header=False, index=False)

Bact_arc_train_profile.to_csv(outdir+"/Bact_arc_train_no_clr.tsv", sep="\t", header=False, index=False)
Bact_arc_val_profile.to_csv(outdir+"/Bact_arc_val_no_clr.tsv", sep="\t", header=False, index=False)
Bact_arc_test_profile.to_csv(outdir+"/Bact_arc_test_no_clr.tsv", sep="\t", header=False, index=False)
Phage_train_profile.to_csv(outdir+"/Phage_train_no_clr.tsv", sep="\t", header=False, index=False)
Phage_val_profile.to_csv(outdir+"/Phage_val_no_clr.tsv", sep="\t", header=False, index=False)
Phage_test_profile.to_csv(outdir+"/Phage_test_no_clr.tsv", sep="\t", header=False, index=False)

with open(outdir + "/Bact_arc_feature_names.txt", "w") as g:
    g.write("\n".join(Bact_arc_train_df.columns.values)+"\n")

with open(outdir + "/Phage_feature_names.txt", "w") as g:
    g.write("\n".join(Phage_train_df.columns.values)+"\n")

with open(outdir + "/train_samples.txt", "w") as g:
    g.write("\n".join(train_samples)+"\n")

with open(outdir + "/val_samples.txt", "w") as g:
    g.write("\n".join(val_samples)+"\n")

with open(outdir + "/test_samples.txt", "w") as g:
    g.write("\n".join(test_samples)+"\n")