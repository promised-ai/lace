from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import polars as pl
import itertools as it


class Dimension:
    Rows = 0
    Colums = 1


FN_IS_SYMMETRIC = {
    'mi': False,
    'depprob': False,
    'rowsim': False,
}


FN_DIMENSION = {
    'mi': Dimension.Colums,
    'depprob': Dimension.Colums,
    'rowsim': Dimension.Rows,
}

def get_all_pairs(fn_name, engine):
    if not fn_name in FN_DIMENSION:
        raise ValueError(f'{fn_name} is an invalid pairwise function')

    if FN_DIMENSION[fn_name] == Dimension.Rows:
        indices = engine.index
    else:
        indices = engine.columns

    symmetric = FN_IS_SYMMETRIC[fn_name]

    if symmetric:
        pairs = []
        n = len(indices)
        for i in range(n):
            for j in range(i, n):
                pairs.append((indices[i], indices[j]))
    else:
        pairs = list(it.product(indices, indices))

    return pairs,  symmetric


def hcluster(df: pl.DataFrame, method='ward'):
    z = linkage(df[:, 1:], method=method, optimal_ordering=True)
    dendro = dendrogram(z, no_plot=True)
    leaves = dendro['leaves']
    col_ixs = [0] + [i + 1 for i in leaves]
    return df[leaves, col_ixs], z

