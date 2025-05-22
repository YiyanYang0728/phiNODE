import pandas as pd
import numpy as np
from skbio.stats.composition import clr

def data_preprocess(infile, pseudocount=1e-20):
    df = pd.read_csv(infile, index_col=0)
    comp_df = pd.DataFrame(data=np.transpose(clr(df.transpose() + pseudocount)), 
                                index=df.index, columns=df.columns)

    comp_df = comp_df.transpose()
    print(comp_df.shape)
    data = comp_df.values
    return(data)
