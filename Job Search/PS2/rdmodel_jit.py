# -*- coding: utf-8 -*-

# @author: Johannes Schmieder
# EC 751 - Labor Economics - Problem Set 1


import sys
import numpy as np # import numpy library
from numba import njit, vectorize # import just in time compiling and vectorization from numba
import pandas as pd
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator



#from output_latex import writeln

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=2)
np.set_printoptions(linewidth=152)

# Flow Utility from consumption v(.)
@njit(cache=True)  # this line enables just in time compilation for the following function. This speeds up the computing
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
@njit(cache=True)  
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
        gainloss = eta * (v(c)-v(r))
    if c<=r:
        gainloss = eta * lam * (v(c)-v(r))
    return v(c) + gainloss

# Search Cost: here arguments are scale parameter (k), level of search effort (s) and curvature parameter (gamma)
@njit(cache=True)
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
    return k * s**(1+gamma)/(1+gamma)

# First Order Condition for Search Effort based on psi function
@njit(cache=True)
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
    # search = min(1,max(delta/k*(Ve-Vu),0)**(1/gamma))
    # The following line is better to avoid overflow errors:
    search = min(1,max(delta/k*(Ve-Vu),0))**(1/gamma)
    # search = 0
    return search

# Steady state FOC for Vu
@njit(cache=True)
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
    s = cost_prime_inverse(Vu,Ve,k,gamma,delta)
    gap = v(benefits[-1])-cost(s,k,gamma)+delta* (s * Ve + (1-s) * Vu) - Vu
    return gap

# Compute V_U steady state using Newton Raphson
@njit(cache=True)
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
    # Newton Raphson Root Finding algorithm
    # Tolerance:
    tolerance=1e-11
    # step for numerical approx of derivative
    h=1e-10
    # Maximum iterations
    maxiter = 100
    # Initial Value for fsolve
    x = Ve

    def f(x):
        return steady_state_FOC(x,Ve,k,gamma,delta,benefits)

    steps_taken = 0
    while (abs(f(x)) > tolerance) & (steps_taken<maxiter):
        # Gradient approx:
        df = (f(x+h) - f(x)) / h
        # Avoid division by zero error
        if df==0:
            break
        x = x-f(x)/df

        steps_taken = steps_taken + 1

    Vu = x

    return Vu # np.array([Vu])

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
    benefits        = np.zeros(T)
    benefits[0:T1]  = b1
    benefits[T1:T2] = b2
    benefits[T2:T3] = b3
    benefits[T3:T]  = welfare

    return benefits

# function to solve individual labor supply model
@njit(cache=True)
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
    T,w_pre,w_post,benefits  = institutions
    T = int(T)

    # Matrix of Income Paths
    # Size of Matrix: (T x T+1)
    # Columns correspond to the Income in the respective period
    # Rows correspond to period j in which job is accepted
    # There is one more row than columns for individuals who never accept a job

    IncomePaths = np.zeros((T+1,len(benefits)))
    for j in np.arange(T+1):
        IncomePaths[j] = benefits

    for j in np.arange(T):
        IncomePaths[j,j:] = w_post

    # Reference Point Paths
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
                    partial = np.array([0.]) # Included in N_pre * w_pre
                else:
                    partial = IncomePaths[:,partial_index] * (N-np.floor(N))

                # r[:,t] = (N_pre*w_pre +
                #         partial +
                #         np.sum(IncomePaths[:,start:end],axis=1,keepdims=False )) / N

                r[:,t] = (N_pre*w_pre +
                        partial +
                        np.sum(IncomePaths[:,start:end],axis=1 )) / N
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
    for t in np.arange(T-int(np.ceil(N))):
        V_E[t] = v(w_post) / (1-delta)
        # Add gain-loss part
        for i in np.arange(int(np.ceil(N))):
            V_E[t] = V_E[t] + eta * delta**i * (v(w_post) - v(r[t,t+i])  )

    # fill in the rest
    for t in np.arange(T-int(np.ceil(N)),T):
        V_E[t] = V_E[T-int(np.ceil(N))-1]

    # Compute Steady State V_U
    V_U[T-1] = steady_state_Vu(V_E[T-1], k, gamma, delta, benefits)

    # print(V_U[T-1])
    # print(steady_state_FOC(V_U[T-1], V_E[T-1],k,gamma,delta,benefits))
    # Steade state search effort
    s[T-1] = cost_prime_inverse(V_U[T-1],V_E[T-1], k,gamma,delta)

    # Backwards Induction (note we start at T-2, since T-1 is last period index)
    for t in np.arange(T-2,-1,-1):
        # print(t)
        s[t] = cost_prime_inverse(V_U[t+1], V_E[t+1], k,gamma,delta)
        # print(IncomePaths[T,t])
        V_U[t] = u(benefits[t],r[T,t],eta,lam) \
                 - cost(s[t],k,gamma)+delta* (s[t]* V_E[t+1] + (1-s[t])*V_U[t+1])

    # Calculate Survival Function
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
    if hasattr(params, "columns") and "value" in params.columns:
        params_vec = params["value"].to_numpy()
    else:
        params_vec = np.asarray(params).ravel()

    params1 = np.copy(params_vec[:6])

    if len(params_vec)==6:
        # Parameters for single type model
        r,s,surv,Vu,Ve = solveSingleTypeModel(params1,institutions)
        return r,s,surv,Vu,Ve
    # Parameters for single type model
    #params1 = np.copy(params[0:6])

    # Variables for 2-type estimation
    elif len(params_vec)==8:
        eta, lam, N, delta, gamma, k1, k2, q1 = params_vec
        shares = np.array([q1, 1-q1])
        kvals = [k1, k2]
        nTypes = 2
    # Variables for 3-type estimation
    elif len(params_vec)==10:
        eta, lam, N, delta, gamma, k1, k2, k3, q1, q2 = params_vec
        shares = np.array([q1, q2, 1-q1-q2] )
        kvals = [k1, k2, k3]
        nTypes = 3

    else:
        raise ValueError('Number of types not supported. Please use 1, 2 or 3 types.')
        

    #k1 = params[5]
    #k2 = params[6]
    #k3 = params[7]
    #kvals = [k1, k2, k3]
    #q1 = params[8]
    #q2 = params[9]
    #shares = np.array([q1, q2, 1-q1-q2] )

    T,w_pre,w_post,benefits  = institutions
    nPeriods = T
    #nTypes = 3

    r      = np.zeros((nPeriods,nTypes))
    s      = np.zeros((nPeriods,nTypes))
    surv   = np.zeros((nPeriods,nTypes))
    V_U    = np.zeros((nPeriods,nTypes))
    V_E    = np.zeros((nPeriods,nTypes))

    for j in range(nTypes):
        params1[5] = kvals[j]
        r[:,j], s[:,j], surv[:,j], V_E[:,j], V_U[:,j] = solveSingleTypeModel(params1,institutions)

    rAgg     = r @ shares
    survAgg  = surv @ shares
    V_EAgg   = V_E @ shares
    V_UAgg   = V_U @ shares

    sAgg = np.zeros(T)
    for t in np.arange(T-1):
        if survAgg[t]>=1e-7:
            sAgg[t] = ( survAgg[t] - survAgg[t+1]) / survAgg[t]

    sAgg[T-1] = sAgg[T-2]

    return rAgg, sAgg, survAgg, V_EAgg, V_UAgg

def drawfig(country,T,lines,labels,lstyle,lcolor,title,figpath,figname,logdir,logfile):
    if country == 'Hungary':
        timevec = np.arange(T) * 15 + 30
        maxT = T-9
    if country == 'Germany':
        timevec = np.arange(T) + 1
        maxT = T-12

    fig = plt.figure()
    plt.clf()
    for i in np.arange(len(lines)):
        xdata = timevec[:len(lines[i])]
        ydata = lines[i]
        xdata = xdata[:maxT]
        ydata = ydata[:maxT]
        plt.plot(xdata, ydata, linestyle=lstyle[i], color=lcolor[i], label=labels[i])

    plt.title(title)
    plt.xlabel('Days')
    plt.legend(bbox_to_anchor=(0.5,-0.05), loc="center", bbox_transform=fig.transFigure, ncol=2)
    plt.legend(loc='lower left')

    if country == 'Hungary':
        plt.axvline(x=97.5, color='grey', linestyle='dashed')
        plt.axvline(x=277.5, color='grey', linestyle='dashed')
        plt.axvline(x=367.5, color='grey', linestyle='dashed')
    if country == 'Germany':
        plt.axvline(x=12, color='grey', linestyle='dashed')
        plt.axvline(x=15, color='grey', linestyle='dashed')

    plt.ylim(bottom=0)

    if (figpath+figname)!='':
        plt.savefig(logdir+figpath+figname, bbox_inches='tight')

    if ((figpath+figname)!='') & (logfile!=''):
        writeln(logdir,logfile,'\\textbf{'+ title + '} \n \n')
        writeln(logdir,logfile, \
                '\\includegraphics[clip=true,trim=0cm 0cm 0cm 0cm,width = 0.68\\textwidth]{' + figpath + figname + '} \n')

    plt.show(block=False)
    # plt.show()
    plt.close(fig)


def simulate_moments(params, institutions_pre, institutions_post):

    # Simulate Model
    r_pre, s_pre, surv_pre, V_U_pre, V_E_pre      = \
        solveMultiTypeModel(params,institutions_pre)

    r_post, s_post, surv_post, V_U_post, V_E_post = \
        solveMultiTypeModel(params,institutions_post)

    # Return Moments
    if institutions_pre[0]==44: # Hungary
        moments_model = np.hstack((s_pre[:35], s_post[:35]))
    if institutions_pre[0]==30: # Germany
        moments_model = np.hstack((s_pre[:18], s_post[:18]))

    return moments_model

def sse(params, target, W, institutions_pre, institutions_post):
    simmoments = simulate_moments(params, institutions_pre, institutions_post)

    # Deviations between target moments and simulated moments:
    err= target - simmoments
    # Calculate SSE
    SSEval = err.T @ W @ err

    return SSEval

def matchingMoments(country):
    if country == 'Hungary':
        momentsfile = './base_moments_Hungary.xlsx'
    if country == 'Germany':
        momentsfile = './base_moments_Germany.xlsx'

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

def getInstitutions(country):

    if country == 'Hungary':
        w_pre = 675
        w_post = 675
        welfare = 90
        b1 = 222
        b2 = 222
        b3 = 114
        T1 = 6
        T2 = 18
        T3 = 24
        T  = T3+20

        ben_pre = benefit_path(b1,b2,b3,welfare,T1,T2,T3,T)

        b1 = 342
        b2 = 171
        ben_post = benefit_path(b1,b2,b3,welfare,T1,T2,T3,T)

    if country == 'Germany':
        w_pre = 1610
        w_post = 1449
        welfare = 400
        b = 1022.35

        T = 30
        P1 = 12
        P2 = 15
        ben_pre = np.ones(T) * welfare
        ben_post = np.ones(T) * welfare
        ben_pre[:P1] = b
        ben_post[:P2] = b

    institutions_pre = T,w_pre,w_post,ben_pre
    institutions_post = T,w_pre,w_post,ben_post

    return institutions_pre, institutions_post


# Create a class for gmm
class gmm:
    # Store inputs inside the object, so later functions in the class can use them later
    # That is we create attributes within the class so that we will only have to call them once.
    # attributes within this class are: iterations, parameters, target (empirical moments), Weighting matrix
    # institutions pre and post reform, and disp that stores whether output shold be displayed during estimation
    def __init__(self, params_full, target, W, institutions_pre, institutions_post, disp=False):
        self.iter = 0
        self.params_full = params_full
        self.target = target
        self.W = W
        self.institutions_pre = institutions_pre
        self.institutions_post = institutions_post
        self.disp = disp

    def sse(self,params):
        # Deviations between target moments and simulated moments:
        # code updates the full parameter vector with the current candiate parameters (parameter values being tested)
        self.params_full.update(params)
        # next line converts the parameter dataframe into a numpy array, which is then used to simulate moments
        params_full_vec = np.array(self.params_full['value'])
        # simulate moments using the current candidate parameter values
        simmoments = simulate_moments(params_full_vec, self.institutions_pre, self.institutions_post)

        # Deviations between target moments and simulated moments:
        # create the error term (^m - m(e)), here e represents parameters
        err= self.target - simmoments
        # Calculate SSE
        # SSE is the is the minimization criterion (minimize distance between empirical and simulated moments)
        # SSE = (^m - m(e))'W(^m - m(e))
        SSEval = err.T @ self.W @ err

        # Update the iteration counter and print the current SSE if disp is True
        # such that each run adds 1 to the iteration counter 
        self.iter = self.iter+1
        if self.disp:
            print('Iter: {:.0f}; Current SSE: {:10.3f}'.format(self.iter, sse))

        return SSEval

    # This function basically uses a cholesky decomposition to rewrite the SSE function above
    # advantages are:
    #1 turns the objective into a standard least-squares form, which many optimizers handle very naturally;
    #2 is numerically stable for symmetric positive-definite W;
    #3 gives you individual weighted residuals (root_contributions), not just the final scalar SSE
    def criterion(self,params):
        # Deviations between target moments and simulated moments:
        # Update full parameter vector with the current candidate parameters (parameter values being tested)
        self.params_full.update(params)
        # Convert parameter dataframe into a numpy array
        params_full_vec = np.array(self.params_full['value'])
        # Simulate moments using the current candidate parameter values
        simmoments = simulate_moments(params_full_vec, self.institutions_pre, self.institutions_post)

        # Deviations between target moments and simulated moments:
        #Create the error term (^m - m(e)), where e represents parameters
        err= self.target - simmoments
        # Calculate SSE
        # SSEval = err.T @ self.W @ err

        # Cholesky decomposition of weighting matrix W (inverse of empirical covariance matrix)
        # Possible for a symmetric and positive definite matrix, which is the case for covariance matrices
        # Turn the symmetic positive definite matrix W into a lower triangular matrix L such that W = L * L^T
        # To gain more speed as it roughly halfs the number of floating-point operations
        # This is done to avoid numerical instability (minimizes rounding errors and eliminates need for row pivoting)
        L = np.linalg.cholesky(self.W)
        # weigh residuals by multiplying the error term with the lower triangular matrix L
        weighted_residuals = err @ L

        # square the weighted residuals to get the contributions to the SSE
        weighted_residuals_squared = weighted_residuals**2

        # sum the squared weighted residuals to get the SSE
        sse = weighted_residuals_squared.sum()
        out = {
            # root_contributions are the least squares residuals.
            # if you square and sum them, you get the criterion value
            "root_contributions": weighted_residuals,
            # if you sum up contributions, you get the criterion value
            "contributions": weighted_residuals_squared,
            # this is the standard output
            "value": sse,
        }
        self.iter = self.iter+1
        if self.disp:
            print('Iter: {:.0f}; Current SSE: {:10.3f}'.format(self.iter, sse))

        return out

    # Similar to the criterion function above, but with added noise to the simulated moments
    # The noise is normal distributed with mean 0 and standard deviation 0.1 in the same units as the moments
    # robustness/stress test of the estimation to see how the estimation performs under noisy data
    # I.E does small random perbutations to the simulated moments lead to the estimated parameters or the criterion value changing drastically
    def criterion_noise(self,params):
        # noise = random.normal(0.1)
        # Deviations between target moments and simulated moments:
        self.params_full.update(params)
        params_full_vec = np.array(self.params_full['value'])
        simmoments = simulate_moments(params_full_vec, self.institutions_pre, self.institutions_post)
        simmoments = simmoments + np.random.normal(0,0.1,size=simmoments.size)
        # Deviations between target moments and simulated moments:
        err= self.target - simmoments
        # Calculate SSE
        # SSEval = err.T @ self.W @ err

        L = np.linalg.cholesky(self.W)
        weighted_residuals = err @ L

        weighted_residuals_squared = weighted_residuals**2
        sse = weighted_residuals_squared.sum()
        out = {
            # root_contributions are the least squares residuals.
            # if you square and sum them, you get the criterion value
            "root_contributions": weighted_residuals,
            # if you sum up contributions, you get the criterion value
            "contributions": weighted_residuals_squared,
            # this is the standard output
            "value": sse,
        }
        self.iter = self.iter+1
        if self.disp:
            print('Iter: {:.0f}; Current SSE: {:10.3f}'.format(self.iter, sse))

        return out
