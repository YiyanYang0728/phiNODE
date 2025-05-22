import sys
import numpy as np
import pandas as pd

Y_delta_path = sys.argv[1]
x_delta_path = sys.argv[2]
outfile = sys.argv[3]

Y = pd.read_table(Y_delta_path, header=None)
sample_n = Y.shape[0]
X = pd.read_table(x_delta_path, header=None)
Y = Y.to_numpy()
x = X.iloc[:,0].to_numpy()

# Z = np.sum(Y/x[:, np.newaxis], axis=0)/sample_n # expected dim: bacterial feature No.
Z = Y/x[:, np.newaxis] # expected dim: sample No. * bacterial feature No.
# print(Z.shape)

df = pd.DataFrame(data=Z)
df.to_csv(outfile, sep="\t", header=False, index=False)