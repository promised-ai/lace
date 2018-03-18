import sys
import os
import pandas as pd

CTGRL_MD = (
    "    - Column:\n"
    "        id: {}\n"
    "        name: {}\n"
    "        colmd:\n"
    "            Categorical:\n"
    "                k: {}\n"
    "                value_map: null\n"
    "                hyper: null\n")

CTS_MD = (
    "    - Column:\n"
    "        id: {}\n"
    "        name: {}\n"
    "        colmd:\n"
    "            Continuous:\n"
    "                hyper: null\n")


class DType:
    categorical = 0
    continuous = 1


def determine_col_type(srs, ctgrl_cutoff=20):
    n_unique = len(srs.unique())
    if n_unique <= ctgrl_cutoff:
        if all(float(int(x)) == x for x in srs.dropna().values):
            k = int(max(srs.dropna().values) + 1)
            dtype = DType.categorical
        else:
            dtype = DType.continuous
            k = 0
    else:
        dtype = DType.continuous
        k = 0

    return dtype, k


def col_metadata(ix, name, dtype):
    if dtype[0] == DType.categorical:
        md = CTGRL_MD.format(ix, name, dtype[1])
    elif dtype[0] == DType.continuous:
        md = CTS_MD.format(ix, name)
    else:
        raise TypeError("Don't know what to do with {}".format(dtype))
    return md


def default_codebook(csv_filename, ctgrl_cutoff=20):
    df = pd.read_csv(csv_filename, index_col=0)
    dtypes = [determine_col_type(df[col], ctgrl_cutoff) for col in df]

    fname = os.path.basename(csv_filename)
    codebook_str = (
        "---\n"
        "table_name: {}\n"
        "metadata:\n"
        "    - StateAlpha:\n"
        "        alpha: 1.0\n"
        "    - ViewAlpha:\n"
        "        alpha: 1.0\n").format(fname)

    for ix, (name, dtype) in enumerate(zip(df.columns, dtypes)):
        codebook_str += col_metadata(ix, name, dtype)

    codebook_str += "rownames:\n"
    for rowname in df.index:
        codebook_str += "    - {}\n".format(rowname)

    return codebook_str


if __name__ == "__main__":
    args = sys.argv[1:]
    helpstr = "Use:\n\t defcb <input.csv> <output.yaml>"

    if args[0] in ["-h", "--help"] or len(args) != 2:
        print(helpstr)
    else:
        csv_filename, yaml_filename = args

    codebook_str = default_codebook(csv_filename)
    with open(yaml_filename, 'w') as f:
        f.write(codebook_str)
        print("Codebook written to %s" % yaml_filename)
