from itertools import combinations

import numpy as np


bins = []
org_bin = [True, True, True, True, True, True]
for idxs in combinations(range(25), 1):
    perturbed_idxs = list(idxs)
    perturbed_bin = org_bin.copy()

    # for idx in perturbed_idxs:
    # perturbed_bin[idx] = not perturbed_bin[idx]

    # print(idxs, " ", perturbed_bin)
    bins.append(perturbed_bin)
print(len(bins))


# print(bins)
