import pickle

def dump(ob, fname):
    with open(fname, 'wb') as f:
        pickle.dump(ob, f, protocol=4)


def load(fname):
    with open(fname, 'rb') as f:
        ob=pickle.load(f)
    return ob