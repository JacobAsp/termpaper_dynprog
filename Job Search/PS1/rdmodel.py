# -*- coding: utf-8 -*-
"""
@author: Johannes Schmieder
Problem Set

"""

import sys
import numpy as np # import numpy library
import pandas as pd
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Flow Utility from consumption v(.)
def v(c): # first line starts with def, then the name of the function and its arguments, and then ":"
    '''
    Returns the flow utility from consumption.
        Arguments:
            c (float): Consumption level
        Returns:conda
            u(c) (float): Utility function evaluated in c.
    '''
    return np.log(c) # log utility

# Per period utility function (with reference dependence)
def u(c,r,eta,lam):
    '''
    Returns the value of the utility function.
        Arguments:
            c (float): Consumption level
            r (float): Reference point
            eta (float): gain loss parameters
            lambda (float): loss aversion parameter
        Returns:conda
            u(c) (float): Utility function evaluated in c.
    '''
    if c>r:
        gainloss = eta*(v((c)) - v(r))
    if c<=r:
        gainloss = eta*lam*(v(c)-v(r))

    return v(c) + gainloss

# Search Cost: here arguments are scale parameter (k), level of search effort (s) and curvature parameter (gamma)
def cost(s,k,gamma):
    '''
    Returns the value of the search cost function.
        Arguments:
            s (float): Search effort
            k (float): Constant parameter in the search cost function
            gamma (float): Shape/elasticity parameter in the search cost function.
        Returns:
            cost(args) (float): Value of the search cost function.
    '''
    return k*s**(1+gamma)/(1+gamma)

# First Order Condition for Search Effort based on psi function
def cost_prime_inverse(Vu,Ve,k,gamma,delta):
    '''
    Returns the value of the inverse of the first derivative of the search cost function.
        Arguments:
            k (float): Scaling parameter in the search cost function
            gamma (float): Shape/elasticity parameter in the search cost function.
            delta (float): Intertemporal discount factor
            valSearch (float): Net value of job search while unemployed
        Returns:
            search (float): Value of the inverse of the first derivative of the search cost function.
    '''
    # Since s is a probability and within [0,1], we set boundaries at 0 and 1, to make sure, it does not return a value outside this interval
    search = min(1,max(delta/k*(Ve-Vu),0))**(1/gamma)
    # search = 0
    return search

# Steady state FOC for Vu
def steady_state_FOC(Vu, Ve, k, gamma, delta, benefits):
    '''
    Returns the value of the gap between the FOC being 0
        Arguments:
            Vu (float): Steady State Value of Unemployment
            Ve (float): Steady State Value of Employment
            k (float): Scaling parameter in the search cost function
            gamma (float): Shape/elasticity parameter in the search cost function.
            delta (float): Intertemporal discount factor
            benefits (array): Benefit path (only last value will be used)
        Returns:
            gap (float): gap for FOC. gap==0 means FOC is holding
    '''
    # With H2M consumption: c_t = y_t, on benefits y_t = b_t, so c_t = b_t
    # The value function of unemployment: Vu_t = v(c_t) - c(s_t) + delta(s_tVe_t+1 + (1-s_t)Vu_t+1)
    # but we can rewrite that as: Vu_t = v(b_t) - c(s_t) + delta(s_tVe_t+1 + (1-s_t)Vu_t+1)
    # In the final tier of benefit system (when benefits doesn't change anymore):
    # then b_t = b_t+1 = b, meaning that Vu_t = Vu_t+1 (steady state condition)
    # we can then write Vu = v(b) - c(s*) + delta(s*Ve + (1-s*)Vu) for this to be zero
    #  0 = v(b) - c(s*) + delta(s*Ve + (1-s*)Vu) - Vu

    # our s* which is the inverse cost function of delta/k(Ve-Vu)
    s = cost_prime_inverse(Vu,Ve,k,gamma,delta)
    # Steady state requires last element of benefit schedule
    #b = benefits[-1]
    # In steady state Vu = v(b) - cost(s, k, gamma) + delta*(s*Ve + (1-s)*Vu)
    # or 0 = (v(b) - cost(s, k, gamma) + delta*(s*Ve + (1-s)*Vu) - Vu)
    # Thus we seek to minimize the gap function
    gap = (v(benefits[-1]) - cost(s, k, gamma) + delta*(s*Ve + (1-s)*Vu) - Vu )
    return gap

# Compute V_U steady state using numerical solver (fsolve)
# This function numerically solves for the minimum of the gap, by inserting our Ve and the parameters of the model and benefits
# Then numerically finding the value for Vu that minimizes the gap function, which is the steady state value of unemployment.
def steady_state_Vu(Ve, k, gamma, delta, benefits):
    '''
    Returns the steady state value of unemployment.
        This is found using fsolve from scipy.optimize.
        Arguments:
            Ve (float): Steady State Value of Employment
            k (float): Scaling parameter in the search cost function
            gamma (float): Shape/elasticity parameter in the search cost function.
            delta (float): Intertemporal discount factor
            benefits (array): Benefit path (only last value will be used)
        Returns:
            Vu (float): Steady State Value of Unemployment
    '''
    # Initial Value for fsolve
    x0 = Ve
    # The next line uses fsolve to find the steady state value of Vu
    Vu = fsolve(steady_state_FOC, x0, args=(Ve, k, gamma, delta, benefits))

    return Vu

# UI benefit level b as function of time
def benefit_path(b1,b2,b3,welfare,T1,T2,T3,T):
    '''
    Returns the benefit path given parameter values.
        Arguments:
            b1,b2,b3,welfare (all floats): values at distinct time windows during unemployment
            T1,T2,T3 (int): points in Unemp. spell when benefit levels change.
            T (int): Total number of periods
        Returns:
            benefits (array): Benefit path in unemployment
    '''
    benefits        = np.zeros(T)       # Laver en vektor med T, elementer. 
    benefits[0:T1]  = b1                # for de rigtige tidspunkter tilføjer den benefit levels. 
    benefits[T1:T2] = b2
    benefits[T2:T3] = b3
    benefits[T3:T]  = welfare

    return benefits

# function to solve individual labor supply model
def solveSingleTypeModel(params,institutions):
    '''
    Solves the retirement model for a single-type individual.
        Arguments:
            params (array): Array of structural parameters.
            institutions (array): Array of institional parameters
        Returns:
            valOLF (array): Value function of OLF

    '''

    eta,lam,N,delta,gamma,k= params
    # eta: gain-loss paramter
    # lam: loss aversion parameter
    # N: number of periods over which reference point is calculated
    # delta: intertemporal discount factor
    # gamma: curvature parameter in search cost function

    T,w_pre,w_post,benefits  = institutions
    # T: number of periods in the model
    # w_pre: previous wage
    # w_post: wage post unemployment
    # benefits: array of benefit levels

    # Matrix of Income Paths
    # Size of Matrix: (T+1 x T)
    # Columns correspond to the Income in the respective period
    # Rows correspond to period j in which job is accepted
    # There is one more row than columns for individuals who never accept a job
    # essentially a matrix where row j represents the income path of an individual who accepts a job in period j, and the columns represent the income received in each period from 0 to T.
    # so for row j, the first j columns will be benefits, while the period after the job is accepted(j+1) will be w_post for the rest of the periods.
    IncomePaths = np.tile(benefits,(T+1,1))
    for j in np.arange(T):
        IncomePaths[j,j:] = w_post

    # Reference Point Paths
    # This creates the reference point matrix, a matrix of same size as the income path matrix (T+1 x T)
    # each row represents the reference point for an individual who accepts a job in period j
    # the columns represent the reference point in each period from 0 to T.
    # if t=0, the reference point is w_pre (agents memory consists only of the previous wage)
    # if t>0, the reference point is calculated as the average of the last N periods of income, where N is a parameter that can be set by the user.
    # thus for N > t, there will be t periods of benefits and N-t periods of w_pre.
    # as the reference point can be part of a period, we need to calculate the partial reference point for the last period of the N periods.
    # let's assume N=5.5 and t=3, then the reference point will be calculated as the average of the last 5.5 periods of income,
    # which consists of 3 periods of benefits and 2.5 periods of w_pre. (reason for flooring N()
    # lastly r_U is the reference point when remaining unemployed, which is the last column of the reference point matrix.
    if eta>0:
        r = np.zeros((T+1,T))
        for t in np.arange(T):
            if t==0:
                r[:,t] = w_pre
            if t>0:
                # Number of periods before UI entry:
                N_pre = max(N-t,0)
                # Range of periods since UI entry
                start = max(t-int(np.floor(N)),0)
                end = t
                partial_index = t-int(np.floor(N))-1
                if partial_index<0:
                    partial = 0 # Included in N_pre * w_pre
                else:
                    partial = IncomePaths[:,partial_index] * (N-np.floor(N))

                r[:,t] = (N_pre*w_pre +
                        partial +
                        np.sum(IncomePaths[:,start:end],axis=1,keepdims=False )) / N
    else:
        r = IncomePaths

    # Reference Point in Unemp:
    r_U = r[T,:]

    # Vectors to store search effort and reservation wage
    s = np.zeros(T)

    V_E   = np.zeros(T)
    V_U   = np.zeros(T)

    # print('N: %6.8f' %N)
    # print('T: %6.8f' %T)
    # print(V_E)

    # Compute V_E(t), where t is period of finding job that starts in next period
    # First step is computing the infinite discounted value of receiving post employment wage: v(w_post) / (1-delta)
    # Second step is capturing the temporary gain from earning wage relative to workers reference point
    # that is if N=2.5, then for period 0, 1, 2 the reference point will still reflect unemployment benefits
    # for period 3, the reference point will be a weighted average of 2 periods of benefits and 0.5 periods of w_post
    # once reference point is fully adapted to w_post then r=w_post and the gain-loss part is zero.
    # This means the wage utility lasts infinitely, but the gain-loss part is only temporary and lasts for N periods.


    for t in np.arange(T-int(np.ceil(N))):
        V_E[t] = v(w_post) / (1-delta)
        # Add gain-loss part
        for i in np.arange(int(np.ceil(N))):
            V_E[t] = V_E[t] + eta * delta**i * (v(w_post) - v(r[t,t+i])  )

    # fill in the rest
    # This part fills the last N periods of V_E with the last fully computed employment value
    # let's say T=10 and N=2.5, then for t in range(7,10), V_E[7], V_E[8], V_E[9] will all be set to V_E[6]
    # since T - [int(np.ceil(N))] -1 = 10 - 3 -1 = 6
    # the reason being that near the end of the finite horizon T, the code cannot evaluate all r[t,t+1]
    # let's say t=8 and i=2, then it request r[8,10], which is out of bounds since the last column is 9. 
    # Thus we just set the last N periods to the last fully computed value.
    for t in np.arange(T-int(np.ceil(N)),T):
        V_E[t] = V_E[T-int(np.ceil(N))-1]

    # Compute Steady State V_U
    # This fills the last element of the unemployment vector, with the steady state value of unemployment 
    # This is computed at the last element of the employment vector and given the parameters of the model and benefit path
    V_U[T-1] = steady_state_Vu(V_E[T-1], k, gamma, delta, benefits)

    # print(V_U[T-1])
    # print(steady_state_FOC(V_U[T-1], V_E[T-1],k,gamma,delta,benefits))
    # Steady state search effort
    # worker has entered stationary welfare regime so V_U[T-1]=V_U[T] and V_E[T-1]=V_E[T]
    s[T-1] = cost_prime_inverse(V_U[T-1],V_E[T-1], k,gamma,delta)

    # Backwards Induction (note we start at T-2, since T-1 is last period index)
    # for the backwards induction, we want to compute the s[T-2] from V_U[T-1] and V_E[T-1], 
    # then we can compute V_U[T-2] from s[T-2], which allows us to compute s[T-3] from V_U[T-2] and V_E[T-2], and so on until we reach the first period.
    # The range starts in T-2, so for V_U and V_E to be a later period, we need to compute them at t+1.
    for t in np.arange(T-2,-1,-1):
        # print(t)
        s[t] = cost_prime_inverse(V_U[t+1],V_E[t+1], k,gamma,delta)
        # print(IncomePaths[T,t])
        # Now V_U[t] = v(b_t)-c(s_t) + delta*(s_t*V_E[t+1] + (1-s_t)*V_U[t+1])
        # here v(b_t) if the flow utility depending on consumption, the reference point, gain-loss parameter and loss aversion parameter.
        # the reference point is index at the T'th row (final row where worker never accept a job) 
        # and the t'th column (the period in which the worker is unemployed)
        # so r[T,t] is the reference point for the worker that never accepted a job during the UI spell, in period t.
        # That is the value of consumption at benefit level in period t, minus cost of search effort in period t,
        # plus discounted future value of being employed in period t+1 with prob. s_t and being unenmployed in period t+1 with prop. 1-s_t
        V_U[t] = u(benefits[t], r[T,t], eta, lam) - cost(s[t], k, gamma) + delta*(s[t]*V_E[t+1] + (1-s[t])*V_U[t+1])

    # Calculate Survival Function
    # The survival function is the probability of remaining uenmployed
    # Creates an array/vector of ones of length T, which is the number of periods in the model
    # Then for each period t, the survival probability is the previous periods survival probability
    # multiplied by the probability of not finding a job in the previous period which is 1-s[t-1]
    # So the first row becomes 1*(1-s[0]), the second row becomes 1*(1-s[0])*(1-s[1]), and so on until the last period T-1.
    # Thus we get the share of workers that remain unemployed in each period t
    survival = np.ones(T)
    for t in np.arange(1,T):
        survival[t] = survival[t-1] * (1-s[t-1])


    return r_U, s, survival, V_U, V_E
 
def solveMultiTypeModel(params,institutions):
    '''
    Solves the retirement model in a multi-type setup.
        Arguments:
            params (array): Array of parameter values for all types.
        Returns:
    '''

    # Parameters for single type model
    params1 = np.copy(params[0:6])

    # Variables for 2-type estimation

    k1 = params[5]
    k2 = params[6]
    k3 = params[7]
    kvals = [k1, k2, k3]
    q1 = params[8]
    q2 = params[9]
    shares = np.array([q1, q2, 1-q1-q2] )

    T,w_pre,w_post,benefits  = institutions
    nPeriods = T
    nTypes = 3

    r      = np.zeros((nPeriods,nTypes))
    s      = np.zeros((nPeriods,nTypes))
    surv   = np.zeros((nPeriods,nTypes))
    V_U    = np.zeros((nPeriods,nTypes))
    V_E    = np.zeros((nPeriods,nTypes))

    for j in [0,1,2]:
        params1[5] = kvals[j]
        r[:,j], s[:,j], surv[:,j], V_E[:,j], V_U[:,j] = solveSingleTypeModel(params1,institutions)

    rAgg     = r @ shares
    survAgg  = surv @ shares
    V_EAgg   = V_E @ shares
    V_UAgg   = V_U @ shares

    sAgg = np.zeros(T)
    for t in np.arange(T-1):
        sAgg[t] = ( survAgg[t] - survAgg[t+1]) / survAgg[t]

    sAgg[T-1] = sAgg[T-2]

    return rAgg, sAgg, survAgg, V_EAgg, V_UAgg

def matchingMoments():
    momentsfile = './base_moments_Hungary.xlsx'
    moments_df  = pd.read_excel(momentsfile, index_col=0)
    moments_hazard_pre = moments_df['before_b'].to_numpy()
    moments_hazard_post = moments_df['after_b'].to_numpy()
    moments_hazard_pre = moments_hazard_pre[1:]
    moments_hazard_post = moments_hazard_post[1:]

    target = np.hstack((moments_hazard_pre,moments_hazard_post))

    # Covariance Matrix:
    sd_pre  = moments_df['before_sd'].to_numpy()[1:]
    sd_post = moments_df['after_sd'].to_numpy()[1:]
    var_pre = sd_pre**2
    var_post = sd_post**2
    var = np.hstack((var_pre,var_post))
    cov = np.eye(len(var))*var
    return target, cov


if __name__ == "__main__":

    print('\n\n\n')
    # Set number of decimals when printing numpy arrays
    np.set_printoptions(linewidth=152)
    np.set_printoptions(precision=2, suppress= True, threshold=sys.maxsize)


    # ========= Problem Set 1 =========
    print('Hello World! ')
