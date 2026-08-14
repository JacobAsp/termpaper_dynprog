# -*- coding: utf-8 -*-
# Author: Johannes Schmieder
# EC 751 - Labor Economics - Problem Set 2 - Part 1

import sys
import os
import numpy as np # import numpy library
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pandas as pd
import time
import scipy.optimize as opt
import estimagic as em

from rdmodel import solveMultiTypeModel, matchingMoments, sse, drawfig, getInstitutions, gmm

if not os.path.isdir('./log/'): # it checks if the log directory specified before exists or not. If not, it creates it.
    os.makedirs('./log/')

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=2)
np.set_printoptions(linewidth=152)

print('\n\n\n')
# Set number of decimals when printing numpy arrays
np.set_printoptions(linewidth=152)
np.set_printoptions(precision=3, suppress= True, threshold=sys.maxsize)

#  ==== Define Colors ====
blue    = tuple(np.array([9, 20, 145]) / 256)
purple  = tuple(np.array([92,  6, 89]) / 256)
fuchsia = tuple(np.array([155,  0, 155]) / 256)
green   = tuple(np.array([0, 135, 14]) / 256)
red     = tuple(np.array([128, 0, 2 ]) / 256)
gray    = tuple(np.array([60, 60, 60 ]) / 256)

# Problem Set 2

# === 2. Calibrate Model ===
# --- 2.1 Simulate Moments ---

country = 'Germany'
momentsfile = './base_moments_Germany.xlsx'
moments_df  = pd.read_excel(momentsfile, index_col=0)
moments_hazard_pre = moments_df['before_b'].to_numpy()
moments_hazard_post = moments_df['after_b'].to_numpy()
moments_hazard_pre = moments_hazard_pre[1:]
moments_hazard_post = moments_hazard_post[1:]

institutions_pre, institutions_post = getInstitutions(country)



# Fill in the rest ...

eta = 0
lam = 0
N = 0
delta = .93
gamma = .35
k1 = 173.1
k2 = 49.6
k3 = 12.9
q1= 0.21
q2 = 0.63
params_multi = np.array([eta, lam, N, delta, gamma, k1, k2, k3, q1, q2])




#def sse_g(val):
#    return sse([eta, lam, N, delta, float(val),k1, k2, k3, q1, q2], target, W, institutions_pre, institutions_post)
#x0 = 0.5
#results = opt.minimize(sse_g, x0, method='L-BFGS-B', bounds=((0.1, 2),), options={'disp': True})
#print(f'Optimal gamma for Germany: {results.x[0]}')