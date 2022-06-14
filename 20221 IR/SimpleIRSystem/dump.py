import pickle as pk


def write(obj, filename):
    with open(filename, "wb") as f:
        pk.dump(obj, f)


def read(filename):
    with open(filename, "rb") as f:
        return pk.load(f)


def pkl_to_txt(p_fname, t_fname):
    obj = read(p_fname)
    if isinstance(obj, dict):
        obj = dict(sorted(obj.items()))
    elif isinstance(obj, list):
        obj = list(sorted(obj))

    with open(t_fname, "w", encoding="utf-8") as f:
        for k, v in obj.items():
            f.write(str(k))
            f.write(" ")
            f.write(str(v))
            f.write("\n")