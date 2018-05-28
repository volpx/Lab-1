#!/usr/bin/env python3
import numpy as np
from uncertainties import ufloat
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from functions import *

def trow_6():
    return int(np.random.rand()*6)+1

def trow_12():
    return int(np.random.rand()*12)+1

def trow_20():
    return int(np.random.rand()*20)+1

def trow_n_time(n,trow):
    out=np.empty(n,dtype=np.int8)
    for i in range(n):
        out[i]=trow()
    return out

def main():
    print(trow_n_time(100,trow_6))

if __name__ == '__main__':
    main()
