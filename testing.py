from sklearn.preprocessing import Normalizer
from collections import defaultdict
import scipy.io
import os
import requests
import numpy as np
import pandas as pd
import utils
import IPython.display as ipd
import matplotlib
import resource
import features as ft
import heapq
from sklearn.metrics.pairwise import pairwise_distances

matplotlib.use('TkAgg')
fma = utils.FreeMusicArchive(os.environ.get('FMA_KEY'))


val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print("AFTER!!!! Process usage: ", val)
