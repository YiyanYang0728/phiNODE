from skbio.stats.composition import clr, clr_inv
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import os, sys

"""
This program is used to disturb a phage feature in the original phage profile and 
predict the corresponding changes in bacetrial profile to 
infer potential interactions between bacteria and phage.
"""

def altered_profile(profile: DataFrame, pos: int, target: Series):
    # replace the whole column at pos with a list called target
    profile.iloc[:, pos] = target
    # check shape and rowwise sum
    # print(profile.shape)
    # print(profile.sum(axis=1).to_list()[:10])
    return profile
    
def perturb_profile(profile: DataFrame, pos: int, mode: str):
    if mode == "mean":
        target = profile.iloc[:, pos].mean() # a series of ncol length; the mean vaue for the features across samples
        target = pd.Series([target] * profile.shape[0])
    elif mode == "std":
        target = profile.iloc[:, pos] + profile.iloc[:, pos].std()
    delta = target - profile.iloc[:, pos]
    dt = delta
    df = pd.concat([dt, target, profile.iloc[:, pos]], axis=1)

    new_profile = altered_profile(profile, pos, target)
    return df, new_profile

if __name__ == "__main__":
    ori_phage_path = sys.argv[1] # original profile
    feature_pos = int(sys.argv[2])
    mode = sys.argv[3]
    outdir = sys.argv[4]
    os.makedirs(outdir, exist_ok=True)
    output = os.path.join(outdir, "ft_"+str(feature_pos)+"_profile.tsv")
    delta_output = os.path.join(outdir, "ft_"+str(feature_pos)+"_delta.tsv")

    ori_phage_profile = pd.read_table(ori_phage_path, header=None)
    delta_df, new_phage_profile = perturb_profile(ori_phage_profile, feature_pos, mode)
    new_phage_profile.to_csv(output, sep="\t", header=False, index=False)
    delta_df.to_csv(delta_output, sep="\t", header=False, index=False)
    print("Write new file in", output, "for feature", feature_pos)
