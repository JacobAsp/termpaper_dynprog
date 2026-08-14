# -*- coding: utf-8 -*-
# Author: Johannes Schmieder
# EC 751 - Labor Economics - Problem Set 2 - Part 2

import sys
import os
import numpy as np # import numpy library
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pandas as pd
import time

import estimagic as em

from rdmodel import benefit_path, solveMultiTypeModel, matchingMoments, sse, simulate_moments, drawfig, getInstitutions, gmm
import rdmodel_jit as fast

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

# === Import Moments ===

country = 'Germany'
momentsfile = './base_moments_Germany.xlsx'
moments_df  = pd.read_excel(momentsfile, index_col=0)
moments_hazard_pre = moments_df['before_b'].to_numpy()
moments_hazard_post = moments_df['after_b'].to_numpy()
moments_hazard_pre = moments_hazard_pre[1:]
moments_hazard_post = moments_hazard_post[1:]

institutions_pre, institutions_post = getInstitutions(country)
eta,lam,N,delta,gamma,k1,k2,k3,q1,q2 = [0,0,0,0.93,0.35,173.1,49.6,12.9,0.21,0.63]
paramsMulti = np.array([eta,lam,N,delta,gamma,k1,k2,k3,q1,q2])

target, cov = matchingMoments(country)
W = np.linalg.inv(cov)
print(W[:5,:5])
T = institutions_pre[0]


# === 4 Estimating All parameters ===

# --- 4.1 GMM Class ---
params_full = pd.DataFrame(
    data={
        "value": [0,0,0,0.93,0.35,173.1,49.6,12.9,0.21,0.63]
    },
    index=["eta", "lam", "N", "delta", "gamma", "k1", "k2", "k3", "q1", "q2"],
)
gmm_object = fast.gmm(params_full.copy(),target, W, institutions_pre, institutions_post, disp=False)

params = pd.DataFrame(
    data={
    "value"       : [0.3], "lower_bound" : [0.01],  "upper_bound" : [2]
    },
    index=["gamma"],
)
print(gmm_object.sse(params))

# --- 4.2 Estimate gamma using Estimagic ---

print('\n === Estimation using Estimagic === \n')
res = em.minimize(
    criterion=gmm_object.sse,
    params=params,
    algorithm="scipy_lbfgsb",
)
print(res.params)

print(gmm_object.params_full)
params_result=np.array(gmm_object.params_full['value'])

r1,s1,surv1,Vu1,Ve1 = solveMultiTypeModel(params_result,institutions_pre)
r2,s2,surv2,Vu2,Ve2 = solveMultiTypeModel(params_result,institutions_post)

drawfig(country,T,[s1, s2, moments_hazard_pre, moments_hazard_post],
    ['Sim. Hazard, P=12', 'Sim. Hazard, P=15',
        'Emp. Hazard, P=12', 'Emp. Hazard, P=15'],
    ['dashed','dashed','solid','solid'],
    [blue,red,blue,red],
    'Exit Hazard from Unemployment - Std. Model','','fig_4_hazardsSD.pdf','./log/','')


#  Now you are on your own ...
