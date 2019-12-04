import sys
from plotter import Plotter
import numpy as np

if __name__ == '__main__':
    path = sys.argv[1]
    idx = int(sys.argv[2])

    pc = np.load(path, 'r')
    print ("{0}/{1}".format(idx, len(pc)))
    Plotter.plot_pc(pc[idx], None, idx, name='real', savefig=False)
