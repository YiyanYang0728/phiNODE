import numpy as np
import pandas as pd
import sys

query_path = sys.argv[1] # original profile (old)
subject_path = sys.argv[2] # perturbed profile (now)
output = sys.argv[3]

query_profile = pd.read_table(query_path, header=None)
subject_profile = pd.read_table(subject_path, header=None)

diff = (subject_profile.to_numpy() - query_profile.to_numpy())
column_means = query_profile.mean(axis=0)
# exp_diff = np.exp(diff)
# weighted_sum = np.sum(query_profile.to_numpy() * exp_diff, axis=0)
# print("exp_diff shape", exp_diff.shape)
# print("weighted_sum shape", weighted_sum.shape)
dt = diff

df = pd.DataFrame(data=dt)
df.to_csv(output, sep="\t", index=False, header=False)
