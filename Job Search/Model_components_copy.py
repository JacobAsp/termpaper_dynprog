########################################################################### 
##################### We import the needed packages #######################
import numpy as np
import pandas as pd
from time import process_time
import matplotlib.pyplot as plt
import scipy
from scipy import interpolate  # type: ignore # Interpolation routines
from matplotlib.gridspec import GridSpec
#from numba import njit, vectorize
from scipy.stats import beta



#########################################################################################
#########################################################################################
# HERE WE MAKE ALL THE FUNCTIONS THAT WE NEED FOR THE CLASSSES 
#########################################################################################

#########################################################################################
# We make the gain-loss function
#@njit(cache=True)
def u(c, r, eta, lmbda):        # returns the utility of current consumption + the gain_loss parameter

    gainloss = np.where(
        c > r,
        eta * (np.log(c) - np.log(r)),
        eta * lmbda * (np.log(c) - np.log(r))
    )

    return np.log(c) + gainloss

#########################################################################################
# We make the search cost function
# Search Cost: here arguments are scale parameter (k), level of search effort (s) and curvature parameter (gamma)
#@njit(cache=True)
def search_cost(s,k,gamma):
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


#########################################################################################
# We make the weights for the asset grid (asset distribution) to be used in the simulation of moments.
# Weights: here arguments are the setting Hand-to-Mouth/Full asset model (htm=1 for true HTM, 0 for full asset model)
# The minimum and maximum asset levels (abar) and the number of asset grid points (n_a)
def make_weights(htm, abar, n_a):

    if htm == 1:
        # True HTM model has only one asset state
        return np.array([1.0])

    else:
        # Full asset model
        a_grid = np.linspace(abar[0], abar[1], n_a)

        # Scale to [0,1]
        x = (a_grid - abar[0]) / (abar[1] - abar[0])        # vi dividerer alle værdier med den højeste værdi. Starter ved 0. 
                                                            # på den måde er vi kun mellem 0 og 1. 
                                                            # Jeg forstår ikke, hvorfor, at det er her er nødvendigt. 

        # Bin boundaries
        edges = np.empty(n_a + 1)
        edges[1:-1] = (x[:-1] + x[1:]) / 2  # for alle indices tager vi gennemsnittet af 0,1; 1,2; 2,3 osv. Hvorfor det?
        edges[0] = 0                        # den første værdi skal være 0
        edges[-1] = 1                       # den sidste værdi skal være én

        alpha = 0.3
        beta_param = 1.7
        # Large density near zero
        # Monotonically declining overall
        # Long right tail

        weights = np.diff(
            beta.cdf(edges, alpha, beta_param)                      # vi laver en cdf med bestemte værdier. alpha = startværdien, beta = udviklingen 
        )

        weights /= weights.sum()

        return weights

#########################################################################################
# We make the function for the optimal search effort, given the value of employment and unemployment
#@njit(cache=True)
def optimal_search_effort(V_emp,V_uemp,k,gamma,delta):
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
    #search = min(1,max(delta/k*(V_emp-V_uemp),0))**(1/gamma)
    # np.clip will clip values outside the interval to the interval edges.
    # here we use the interval between 0 and 1, and clip the value of delta/k*(V_emp-V_uemp) to be within this interval.
    search = np.clip(delta/k*(V_emp-V_uemp), 0, 1)**(1/gamma)

    # search = 0
    return search

#########################################################################################
# We make the function for the value of employment using ref path and steady state values 
#@njit(cache=True)
def employment_BI(Vss, ref_path, delta, eta, lmbda, R, w, abar, T, N, n_a, n_c, htm):

    # ---------------------------------------------------------------------
    # Hand-To-Mouth Model: 
    # ---------------------------------------------------------------------
    if htm == 1:
        #Only one asset state: A = 0
        V_emp = np.zeros((1, T+1))   # placeholder variable for getting the job in every period that is not the first period.
        c_emp = np.zeros((1, T+1))   # associated optimal consumption, first period after job start, not really that relevant though

        Vss_htm = np.array(Vss).reshape(-1)[0]  # steady state value from the value function iterations.
        for j in np.arange(T+1):
            V0 = Vss_htm   # steady state value from the value function iterations.
            for n in np.arange(N-1, -1, -1):
                c_current = w
                V0 = u(c_current, ref_path[j,j+n], eta, lmbda) + delta * V0     # i periode T tages SS værdien. Derefter tages nytten af forbrug = w ift. ref path i j'te periiode indtil j+n'te period
                                                                                # fordi man får et job i j'te periode og har nytte fra ref-path i n perioder indtil konvergens. 
                V_emp[0, j] = V0        # tilføjes til matrix. Den bliver 1 x T
                c_emp[0, j] = w
        return V_emp, c_emp

    # ---------------------------------------------------------------------
    # Model with Assets
    # ---------------------------------------------------------------------
    else:
        abar[0] = np.maximum(np.finfo(float).eps, abar[0])              # vælger numerisk nul, så vi kan tage log. 
        a = np.linspace(abar[0], abar[1], n_a).reshape((n_a, 1))        # laver en n_a x 1 matrice med værdierne fra 0 til maks asset level. 

        c = np.empty((n_a, n_c))    # For hver state kan man vælge n_c forskellige niveauer af forbrug i perioden
        for i in range(n_a):        # vi kigger på alle rækkerne og tilføjer forbrugsmuligheder fra 0 og op til maks forbrug som er givet ved assets + løn.  
            c[i, :] = np.linspace(abar[0], (a[i]+w), n_c).reshape((1, n_c))     # Laver det om til en vektor med 1 række og n_c kolonner. 
                                                                                # og det gør vi så n_a gange, så det bliver n_a til n_c
        a1 = (a - c + w) * (1 + R)   # assets i næste periode. Det er en n_a x n_c matrix. 
                                     # den har altså for alle asset levels (rækker) alle muligheder for forbrug (kolonner). 

        V_emp = np.zeros((T+1, n_a))   # placeholder variable for getting the job in period in period T with asset value a. 
        c_emp = np.zeros((T+1, n_a))   # associated optimal consumption, first period after job start, not really that relevant though...

        for j in np.arange(T+1):
            V0 = Vss   # steady state value from the value function iterations. 

            # step backward: n = N-1 (period closest to steady state) down to n = 0 (period job starts)
            for n in np.arange(N-1, -1, -1):
                interp = interpolate.interp1d(a[:, 0], V0, bounds_error=False, fill_value="extrapolate")    # here it always takes the latest V0 value, so a new one in every period. This is where the updated values get used. 
                V_next = interp(a1)     # V_next er jo altid den seneste værdi. Første del af loopet er så den 'interpolerede' værdi fra v_ss af givne asset levels mellem grids. 

                matV1 = u(c,ref_path[j,j+n],eta, lmbda) + delta * V_next    # nytteværdi for givet 'j' jobtidspunkt for n-perioder efter job er fundet.  
                                                                # og så fordi at c er defineret som alle forbrugsmuligheder med forskellige asset levels overholdet det budget begrænsningen. 
                i_max = np.argmax(matV1, axis=1)                # hvordan overholder vi budgetbegrænsningen i næste periode? 
                V0 = matV1[np.arange(n_a), i_max]               # her opdateres værdierne givet det optimale forbrugsval. Det bliver så genbrugt til næste periode. 
                c_current = c[np.arange(n_a), i_max]

            V_emp[j, :] = V0
            c_emp[j, :] = c_current

        V_emp = V_emp.transpose( )
        c_emp = c_emp.transpose()
        return V_emp, c_emp

# I think that there might be a mistake here, because we don't ever update the asset level here. 
# So, it's like a light version of the 'mistake' we've had made before we're we don't forward solve. 
# this basically means that the value of getting the job in period j with A assets is the value of keeping those assets forever
# but acting as if you didn't keep them in every period. 
# this can be fixed if we also forward solve 'employment BI'. 



#########################################################################################
#########################################################################################
# HERE WE MAKE ALL THE CLASSES NEEDED TO SOLVE THE MODEL
#########################################################################################


#########################################################################################
# Class for determining the reference path through the income path/benefit path

class ref_inc_path():
    def __init__(self,w =15, b1=10, b2 = 8, b3 = 6, welfare = 4, T1 = 12, T2 = 36, T3 = 48, T = 54, N = 5, eta = 1 ):
        self.w = w              # Løn
        self.b1 = b1            # højeste dagpengesats    
        self.b2 = b2            # næste dagepengesats
        self.b3 = b3            # laveste dagpengesats
        self.welfare = welfare  # 'kontanthjælp'
        self.T1 = T1            # periode til højeste dagpengesats
        self.T2 = T2            # periode til næste dagpengesats
        self.T3 = T3            # periode til laveste dagpengesats
        self.T = T              # periode til kontanthjælp
        self.N = N              # antal referenceperioder
        self.eta = eta          # det er for at kunne modificere løsningen til også at være ikke reference-dependent

    #@njit(cache=True)
    def benefit_path_(self):                # vi laver income paths som en T, array med alle income levels for forskellige perioder
        '''
        Returns the benefit path given parameter values.
            Arguments:
                b1,b2,b3,welfare (all floats): values at distinct time windows during unemployment
                T1,T2,T3 (int): points in Unemp. spell when benefit levels change.
                T (int): Total number of periods
            Returns:
                benefits (array): Benefit path in unemployment
        '''
        benefits        = np.zeros(self.T)       # Laver en vektor med T, elementer. 
        benefits[0:self.T1]  = self.b1                # for de rigtige tidspunkter tilføjer den benefit levels. 
        benefits[self.T1:self.T2] = self.b2
        benefits[self.T2:self.T3] = self.b3
        benefits[self.T3:self.T]  = self.welfare

        return benefits
    #@njit(cache=True)
    def income_path_(self):                          
        '''Laver alle de mulige income paths. 
            En for alle tidspunkter hvorpå man finder et job, inklusiv muligheden for aldrig at finde et job
        '''
        benefits = self.benefit_path_()
        income_path = np.tile(benefits, (self.T+1,1))

        for j in np.arange(self.T): 
            income_path[j,j:] = self.w
        return income_path
    #@njit(cache=True)
    def ref_path_(self):             # the reference point is given as the arithmetic average of the income in the 5 periods leading up to the current one. 
        if self.eta > 0: 
            ref_path = np.zeros((self.T+1, self.T))
            income_path = self.income_path_()
        
            for t in np.arange(self.T): 
                if t == 0:
                    ref_path[:,t] = self.w         # vi antager at lønnen før og lønnen efter bistand er den samme. 

                if t > 0: 
                    N_pre = max(self.N - t, 0)  # vi sætter antal perioder med løn til at blive mindre, når man har været længere tid på DP. Indtil løn slet ikke indgår i referencen. 
                    '''Her modificerer jeg koden ned i kompleksitet, så vi ikke tager højde for eventuelle halve perioder. 
                    Da vi slet ikke har dem med. Se rdmodel.py, for den komplekse udgave'''
                    start = max(t-self.N, 0)
                    end = t
                        #Her laver vi summen / gennemsnittet. 
                        #Når t > 4 så betyder N_pre leddet ikke længere noget.
                    ref_path[:,t] = ( N_pre*self.w + np.sum(income_path[:,start:end], axis = 1, keepdims=False ) ) / self.N
            
                    # reference path tager gennemsnittet for de seneste fem perioder af indkomst for alle rækker i income path.
                    # så altid summen af de t-5 til t søjler i income path plus evt. led fra før dagpenge, hvis vi er i t < 5
                    # axis = 1 betyder at vi kigger på søjler keepdims = False betyder at vi kun returner en vektor med t værdier pr. række.
                    # så hver række i income path repræsenterer, hvornår man får job. Hver række i r repræsenterer den dertilhørende lønreference i alle perioder. 
        else: 
            ref_path = self.income_path_()    
            
        return ref_path
         
    #@njit(cache=True)    
    def ref_path_long_(self):        # we create this to get the future values of ref-dependence when getting a job inside T-N last periods.
            
        add = np.zeros( (self.T+1, self.N)  )

        ref_path = self.ref_path_()
        ref_path_long = np.concatenate((ref_path, add), axis = 1)      # now we add N columns to the ref_path_
            
        income_path = self.income_path_()       # we add the income path to use it later

        for t in np.arange(self.T, self.T + self.N):
                
            start = min ( t - self.N , self.T)      # vi definerer start som den periode vi er i og minus de N perioder vi kigger tilbage. Vi går indtil sidste periode i income_path
            end = self.T                            # det her er måske overflødigt, fordi vi bare siger, at vi stopper ved enden. 

            ref_path_long[:,t] = (  np.sum(income_path[:,start:], axis = 1, keepdims=False     )   + (t-self.T)* self.w ) / self.N 
                                        # so what the code does is that it takes the sum of the last periods available in the income path
                                        # plus the sum of the periods not available and divides by N to get the average.                                  
        """ændrer den sidste række til blot at være den samme som den var før men 5 perioder mere, fordi han aldrig får et job. """
        
        ref_path_long[-1,self.T] = self.welfare
        return ref_path_long


class ss_value:
    """Class to implement the stochastic cake eating model with discretized choice"""
    def __init__(self, delta=0.9, eta = 1, lmbda = 4.9, abar=[0, 20], n_a=50, n_c=100, w=15, R = 0.05, htm = 0, ):
        """Initializer"""
        self.delta = delta    # Discount factor
        self.eta = eta      # gain-loss value parameter
        self.lmbda = lmbda  # loss aversion parameter
        self.abar = abar    # min and max savings
        self.n_a = n_a      # grid size for state variable
        self.n_c = n_c      # grid size for choice grid
        self.w = w          # wage
        self.R = R          # interest rate
        self.htm = htm

        # ---------------------------------------------------------------------
        # Hand-To-Mouth Model: 
        # ---------------------------------------------------------------------
        if self.htm == 1:
            # Hand to mouth: Only one asset state - A = 0
            self.n_a = 1
            self.n_c = 1

            self.a = np.array([[0.0]])
            self.c = np.array([[self.w]])
        # ---------------------------------------------------------------------
        # Model with Assets: 
        # ---------------------------------------------------------------------
        else:
            self.abar[0] = np.maximum(np.finfo(float).eps, self.abar[0])
            self.a = np.linspace(self.abar[0], self.abar[1], n_a).reshape((n_a, 1))  
            self.c = np.empty((n_a, n_c))
            for i in range(n_a):            # fordi rækkerne er state-grids, så derfor skal vi have en række for hver state vi vil kigge på
                self.c[i, :] = np.linspace(self.abar[0], self.a[i,0]+self.w, n_c) #.reshape((1, n_c)) 
                    
    #@njit(cache=True)
    def bellman(self, V0, R):   # V0 is a vector with guessing initial values. Must be of correct dimensions, but values don't matter
        # ---------------------------------------------------------------------
        # Hand-To-Mouth Model: 
        # ---------------------------------------------------------------------
        if self.htm == 1:
            # No asset choice.
             # A_t = A_{t+1} = 0 and c_t = income.
            c1 = np.array([self.w])

            V1 = np.array([u(self.w,self.w,self.eta,self.lmbda)/ (1-self.delta)])
            return V1, c1
        # ---------------------------------------------------------------------
        # Model with Assets: 
        # ---------------------------------------------------------------------
        else: 
            interp = interpolate.interp1d(self.a[:, 0], V0, bounds_error=False, fill_value="extrapolate")
            a1 = (self.a - self.c + self.w)*(1+R)
            V = interp(a1)

            matV1 = u(self.c, self.w, self.eta, self.lmbda ) + self.delta * V
            i_max = np.argmax(matV1, axis=1)
            V1 = matV1[np.arange(self.n_a), i_max]
            c1 = self.c[np.arange(self.n_a), i_max]

            return V1, c1

        

#@njit(cache=True)
def vfi(self, maxiter=1000, tol=1e-8, callback=None): # machine precision 1e-15
    """Solves the model using VFI (successive approximations)"""
    # ---------------------------------------------------------------------
    # Hand-To-Mouth Model: 
    # ---------------------------------------------------------------------
    if self.htm == 1:
        c1 = np.array([self.w])

        V1 = np.array([
            u(
                self.w,
                self.w,
                self.eta,
                self.lmbda
            ) / (1 - self.delta)
        ])

        return V1, c1
    
    # ---------------------------------------------------------------------
    # Model with Assets: 
    # ---------------------------------------------------------------------

    tic = process_time()  # Start the stopwatch / counter

    V0 = np.log(self.a[:, 0])  # on first iteration assume consuming everything
    
    for iter in range(maxiter):
        V1, c1 = self.bellman(V0, self.R)
        if callback:
            callback(iter, self.a, V1, c1)  # callback for making plots
        if np.all(abs(V1 - V0) < tol):
            toc = process_time()  # Stop the stopwatch / counter
            #uncomment this if you want to see how long it takes to solve the model
            #print("Optimal consumption of assets solved in", iter, "iterations, using", round(toc - tic, 5), "seconds")
            break
        V0 = V1
    else:  # when i went up to maxiter
        print("No convergence: maximum number of iterations achieved!")
    return V1, c1

# Add the vfi_solve method to the cake_stochastic class
ss_value.solve = vfi



#########################################################################################
#########################################################################################
# HERE WE MAKE THE FUNCTION THAT SOLVES THE MODEL GIVEN PREVIOUS FUNCTIONS AND CLASSES
# We solve primarily for value of unemployment and search effort
#########################################################################################

#########################################################################################
# This Function solves the model by backward induction, given the parameters and institutions.
# We then obtain optimal policies for consumption and search effort, as well as the value of being employed and unemployed.
# That is given we are in period t with asset level a.
#########################################################################################
#def SolveModel(delta, gamma, eta, k, lmbda, abar, n_a, n_c, T1, T2, T3, T, N, b1, b2, b3, welfare, w, R ):
#@njit(cache=True)
def SolveModel(params, institutions, abar, htm):
    '''
    Returns the value of the optimal search effort.
        Arguments:
            
            k (float): Scaling parameter in the search cost function
            gamma (float): Shape/elasticity parameter in the search cost function.
            delta (float): Intertemporal discount factor
            valSearch (float): Net value of job search while unemployed
        Returns:
            search (float): Value of the optimal search effort.
    '''
    # Unpack parameters and institutions
    delta, gamma, eta, k, lmbda, N = params
    n_a, n_c, T1, T2, T3, T, b1, b2, b3, welfare, w, R = institutions

    # Integer parameters
    N = int(N)
    n_a = int(n_a)
    n_c = int(n_c)
    T1 = int(T1)
    T2 = int(T2)
    T3 = int(T3)
    T = int(T)
    

    # Effective dimensions of the model
    if htm == 1:
        n_a_eff = 1
        n_c_eff = 1
    else: 
        n_a_eff = n_a
        n_c_eff = n_c

    #Steady State Values for employment and unemployment
    emp = ss_value(delta, eta, lmbda, abar.copy(), n_a_eff, n_c_eff, w, R, htm)
    Vss_emp, css_emp = emp.solve()
    uemp = ss_value(delta, eta, lmbda, abar.copy(), n_a_eff, n_c_eff, welfare, R, htm)
    Vss_uemp, css_uemp = uemp.solve()


    # Income/reference paths
    model = ref_inc_path(w, b1, b2, b3, welfare, T1, T2, T3, T, N, eta )
    ref_path_long = model.ref_path_long_()
    benefits = model.benefit_path_()

    #Employment Value
    V_emp, c_emp = employment_BI(Vss_emp , ref_path_long, delta, eta, lmbda, R, w, abar, T, N, n_a_eff, n_c_eff, htm)

    # Allocate uenmployment objects
    V_uemp = np.zeros((n_a_eff, T+1))   # placeholder variable for getting the job in period in period T with asset value a. 
    c_uemp = np.zeros((n_a_eff, T+1))   # associated optimal consumption, first period after job start, not really that relevant though...
    S = np.zeros((n_a_eff, T+1))   # placeholder variable for optimal search effort in period t with asset value a.

    # Terminal Condition
    V_uemp[:, T] = Vss_uemp # Once we reach T periods, the value of being uemployed is equal to the steady state value of being unemployed.
    c_uemp[:, T] = css_uemp # Once we reach T periods, the optimal consumption is equal to the steady state consumption of being unemployed.
    S[:, T] = 0 # One we reach T searching is no longer possible

        # ---------------------------------------------------------------------
        # Hand-To-Mouth Model: 
        # ---------------------------------------------------------------------
    if htm == 1:
            for t in np.arange(T-1, -1, -1):
                y = benefits[t]  # Current unemployment income

                c_current = y

                V_next_emp = V_emp[0, t+1]  # Successful search: Agent finds a job in next period
                V_next_uemp = V_uemp[0, t+1]  # Unsuccessful search: Agent does not find a job in next period
                s = optimal_search_effort(V_next_emp, V_next_uemp, k, gamma, delta)  # optimal search effort given the value functions for employment and unemployment
                utility_current = u(c_current, ref_path_long[-1, t], eta, lmbda)
                continuation = delta * (s * V_next_emp + (1 - s) * V_next_uemp)
                V_uemp[0, t] = utility_current - search_cost(s, k, gamma) + continuation
                c_uemp[0, t] = c_current
                S[0, t] = s
         
        # ---------------------------------------------------------------------
        # Model with Assets: 
        # ---------------------------------------------------------------------
    else: 

        # Asset grid
        # local abar copy to avoid modifying the original abar
        abar_local = np.array(abar, dtype=float).copy()
        abar_local[0] = np.maximum(np.finfo(float).eps, abar_local[0])              # vælger numerisk nul, så vi kan tage log. 
        a = np.linspace(abar_local[0], abar_local[1], n_a_eff).reshape((n_a_eff, 1))        # laver en n_a x 1 matrice med værdierne fra 0 til maks asset level. 
        
        for t in np.arange(T-1, -1, -1):
            # Current unemployment income
            y = benefits[t]

            # consumption possibilites
            c = np.empty((n_a_eff, n_c_eff))    # For hver state kan man vælge n_c forskellige niveauer af forbrug i perioden
            for i in range(n_a_eff): 
                       # vi kigger på alle rækkerne og tilføjer forbrugsmuligheder fra 0 og op til maks forbrug som er givet ved assets + løn.
                c_min = max(np.finfo(float).eps, a[i, 0] + y - abar_local[1] / (1 + R))  # minimum consumption level to avoid negative assets in the next period
                                                                                         # we maximize over a small positive number to avoid numerical issues with log(0)
                                                                                         # and then the boundary c_t >= A_t + y_t - A_t+1/(1+R) ensures that we don't consume more than what we have in assets + current income - the maximum asset level in the next period.
                c_max = a[i, 0] + y # maximum consumption A can't be negative, so we can't consume more than what we have in assets + current income  
                c[i, :] = np.linspace(c_min, c_max, n_c_eff) #.reshape((1, n_c))     # Laver det om til en vektor med 1 række og n_c kolonner.
                #c[i, :] = np.linspace(abar_local[0], (a[i,0]+y), n_c_eff) #.reshape((1, n_c))     # Laver det om til en vektor med 1 række og n_c kolonner.

            a1 = (a - c + y) * (1 + R)   # assets i næste periode

            

            # Successful search: Agent finds a job in next period
            interp_emp = interpolate.interp1d(a[:, 0], V_emp[:, t+1], bounds_error=False, fill_value="extrapolate")
            V_next_emp = interp_emp(a1)   

            # Unsuccessful search: Agent does not find a job in next period
            interp_uemp = interpolate.interp1d(a[:, 0], V_uemp[:, t+1], bounds_error=False, fill_value="extrapolate")
            V_next_uemp = interp_uemp(a1)  # V_next er jo altid den seneste værdi. Første del af loopet er så den 'interpolerede' værdi fra v_ss af givne asset levels mellem grids.

            # Optimal search effort: Solve for optimal search effort given the value functions for employment and unemployment
            S_choices = optimal_search_effort(V_next_emp, V_next_uemp, k, gamma, delta)  # optimal search effort given the value functions for employment and unemployment

            #Utility for every possibly consumption choice given
            
            # Set last row to never employed reference path
            

            utility_current = u(c, ref_path_long[-1, t], eta, lmbda)
            #np.log(c) + eta * lmbda * (np.log(c) - np.log(reference))

            continuation = delta * (S_choices * V_next_emp + (1 - S_choices) * V_next_uemp)

            matV = utility_current - search_cost(S_choices, k, gamma) + continuation

            # Choose optimal consumption and assets

            i_max = np.argmax(matV, axis=1)               
            V_uemp[:, t] = matV[np.arange(n_a_eff), i_max]               # her opdateres værdierne givet det optimale forbrugsval. Det bliver så genbrugt til næste periode.
            c_uemp[:, t] = c[np.arange(n_a_eff), i_max]
            S[:, t] = S_choices[np.arange(n_a_eff), i_max]

    # Calculate Survival Function
    survival = np.ones((n_a_eff,T+1))
    for t in np.arange(1,T+1):
        survival[:,t] = (survival[:,t-1] * (1-S[:,t-1]) )

    return S, V_emp, V_uemp, c_emp, c_uemp, Vss_emp, Vss_uemp, css_emp, css_uemp, survival, benefits

#########################################################################################
# This Function solves the model forwads by taking the optimal consumption and search effort policies and 
# simulating the evolution of assets, consumption, and search effort over time.
#########################################################################################

def SolveForward_normal(params, institutions, abar, htm):
    # Solve model by backwards induction
    S, V_emp, V_uemp, c_emp, c_uemp, Vss_emp, Vss_uemp, css_emp, css_uemp, survival, benefits =SolveModel(params, institutions, abar, htm)

    #unpack parameters and institutions
    delta, gamma, eta, k, lmbda, N = params
    n_a, n_c, T1, T2, T3, T, b1, b2, b3, welfare, w, R = institutions

    # Cast institution parameters to integers
    n_a = int(n_a)
    n_c = int(n_c)
    T1 = int(T1)
    T2 = int(T2)
    T3 = int(T3)
    T = int(T)

    # Effective dimensions of the model based on whether it is a hand-to-mouth model or not
    if htm == 1:
        n_a_eff = 1
        n_c_eff = 1
    else:
        n_a_eff = n_a
        n_c_eff = n_c
    
    cons = np.empty((n_a_eff, T))
    search = np.empty((n_a_eff, T))
    assets = np.empty((n_a_eff, T + 1))

    # Asset grid
    if htm == 1:
        asset_grid = np.array([0.0])
    else:
        abar_local = np.array(abar, dtype=float).copy()
        abar_local[0] = np.maximum(np.finfo(float).eps, abar_local[0])
        asset_grid = np.linspace(abar_local[0], abar_local[1], n_a)

    # initial assets
    assets[:, 0] = asset_grid



    for i in range(n_a_eff):

        asset_now = float(asset_grid[i])

        for t in range(T):
            if htm == 1:
                c_now = c_uemp[0, t]
                s_now = S[0, t]

                asset_next = 0.0

            else:
                # Consumption policy function c_t(A)
                interp_c = interpolate.interp1d(asset_grid,
                    c_uemp[:, t],
                    bounds_error=True
                )
                # Search policy s_t(A)
                interp_s = interpolate.interp1d(
                    asset_grid,
                    S[:, t],
                    bounds_error=True
                )

                c_now = float(interp_c(asset_now))
                s_now = float(interp_s(asset_now))

                asset_next = (asset_now - c_now + benefits[t]) * (1 + R)
                asset_next = max(asset_next, asset_grid[0])
                if asset_next > asset_grid[-1]:
                    raise ValueError(
                        f"Forward assets exceed asset grid: "
                        f"i={i}, t={t}, "
                        f"A_t={asset_now:.4f}, "
                        f"c_t={c_now:.4f}, "
                        f"y_t={benefits[t]:.4f}, "
                        f"A_next={asset_next:.4f}, "
                        f"A_max={asset_grid[-1]:.4f}, "
                        f"params={params}"
                    )

            # Store results for both hand-to-mouth and asset models
            cons[i, t] = c_now
            search[i, t] = s_now
            assets[i, t + 1] = asset_next
            asset_now = asset_next    

    survival = np.ones((n_a_eff, T + 1))

    for t in range(T):
        survival[:, t + 1] = (
            survival[:, t]
            * (1 - search[:, t])
        )

    return cons, search, survival, assets, asset_grid, benefits,

    #return cons, search, assets, asset_grid

# this model uses the optimal decision rules but alters them in every period with a naive present biased agent. 
def SolveForward_pb(params, institutions, abar, htm, beta = 1.0):
    # Solve model by backwards induction
    S, V_emp, V_uemp, c_emp, c_uemp, Vss_emp, Vss_uemp, css_emp, css_uemp, survival, benefits =SolveModel(params, institutions, abar, htm)

    #unpack parameters and institutions
    delta, gamma, eta, k, lmbda, N = params
    n_a, n_c, T1, T2, T3, T, b1, b2, b3, welfare, w, R = institutions

    # Cast institution parameters to integers
    n_a = int(n_a)
    n_c = int(n_c)
    T1 = int(T1)
    T2 = int(T2)
    T3 = int(T3)
    T = int(T)

    # Effective dimensions of the model based on whether it is a hand-to-mouth model or not
    if htm == 1:
        n_a_eff = 1
        n_c_eff = 1
    else:
        n_a_eff = n_a
        n_c_eff = n_c
    
    cons = np.empty((n_a_eff, T))
    search = np.empty((n_a_eff, T))
    assets = np.empty((n_a_eff, T + 1))

    # Asset grid
    if htm == 1:
        asset_grid = np.array([0.0])
    else:
        abar_local = np.array(abar, dtype=float).copy()
        abar_local[0] = np.maximum(np.finfo(float).eps, abar_local[0])
        asset_grid = np.linspace(abar_local[0], abar_local[1], n_a)

    # initial assets
    assets[:, 0] = asset_grid

    # vi tiføjer ref_path_igen da vi skal genberegne optimale forbrugsvalg. 
    model = ref_inc_path(w, b1, b2, b3, welfare, T1, T2, T3, T, N, eta)
    ref_path_long = model.ref_path_long_()

    c = np.empty((n_a_eff, n_c_eff))    # consumption placeholder variable
    for i in range(n_a_eff):

        asset_now = float(asset_grid[i])

        for t in range(T):          
            # We re-calculate the paths by adding a beta term to optimal search effort in every period. 
            # but we still use the VE and VU found from the optimal decision rule since it's a naive pb agent. 
            # this means that we need to create Vnext_emp and uemp in every period from the optimal decision rules found in solve_model
            # and use them in finding the optimal search effort and consumption once again in every period. 

            if htm == 1:
                V_next_emp = V_emp[0, t+1]
                V_next_uemp = V_uemp[0, t+1]

                c_now = c_uemp[0, t]   
                s_now = optimal_search_effort(V_next_emp, V_next_uemp, k, gamma, beta*delta)  
                

                asset_next = 0.0

            else:           # if assets then present bias also affects consumption smoothing to be more front loaded. We need to re-calculate that
                # define consumption (the same way as when we solve the model)
                y = benefits(t)
                c_min = max(np.finfo(float).eps, asset_now[i, 0] + y - abar_local[1] / (1 + R)) # asset[i] because we solve for every initial asset level
                c_max = asset_now[i, 0] + y

                c[i,:] = np.linspace(c_min, c_max, n_c_eff) # c bliver 1 x n_c_eff matrix med forbrugsmuligheder fra 0 til maks

                a1 = (asset_now - c + y) * (1 + R)

                # Vi interpolerer over mulige fremtidige v_emp og uemp givet nyt forbrugsvalg ved present bias (pb)
                # vi bruger allerede løste v_emp og v_uemp pga. naivitet
                interp_emp = interpolate.interp1d(asset_grid, V_emp[:, t+1], bounds_error=False, fill_value="extrapolate")
                interp_uemp = interpolate.interp1d(asset_grid, V_uemp[:, t+1], bounds_error=False, fill_value="extrapolate")
                
                # finder næste værdi, som er lavet ud fra de gamle værdier. 
                V_next_emp = interp_emp(a1)
                V_next_uemp = interp_uemp(a1)

                # vi bruger dem til at lægge ind i optimal search effort. 
                # her vælger han mht. naive fremtidige værdier, men vidende at han er pb i nuværende periode. 
                S_choices = optimal_search_effort(V_next_emp, V_next_uemp, k, gamma, beta*delta)

                # det her er koden fra når vi løser med BI, men nu løser vi fremad med de allerede fundne v_emp og v_uemp. 
                utility_current = u(c, ref_path_long[-1, t], eta, lmbda)
            
                # her tilføjer vi blot et lille bitte beta led, som han altid glemmer, fordi vi har bestemt v_emp og v_uemp
                continuation = beta * delta * (S_choices * V_next_emp + (1 - S_choices) * V_next_uemp)

                matV = utility_current - search_cost(S_choices, k, gamma) + continuation

                # Vi fjerner v_emp fra original kode fordi den har vi allerede

                i_max = np.argmax(matV, axis=1)               
                c_now[:, t] = c[np.arange(n_a_eff), i_max]
                s_now[:, t] = S_choices[np.arange(n_a_eff), i_max]
                asset_next = a1[i_max]

            
                asset_next = max(asset_next, asset_grid[0])
                if asset_next > asset_grid[-1]:
                    raise ValueError(
                        f"Forward assets exceed asset grid: "
                        f"i={i}, t={t}, "
                        f"A_t={asset_now:.4f}, "
                        f"c_t={c_now:.4f}, "
                        f"y_t={benefits[t]:.4f}, "
                        f"A_next={asset_next:.4f}, "
                        f"A_max={asset_grid[-1]:.4f}, "
                        f"params={params}"
                    )

            # Store results for both hand-to-mouth and asset models
            cons[i, t] = c_now
            search[i, t] = s_now
            assets[i, t + 1] = asset_next
            asset_now = asset_next    

    survival = np.ones((n_a_eff, T + 1))

    for t in range(T):
        survival[:, t + 1] = (
            survival[:, t]
            * (1 - search[:, t])
        )

    return cons, search, survival, assets, asset_grid, benefits,

def SolveForward(params, beta, institutions, abar, htm):
    if beta == 1: 
        SolveForward_normal(params, institutions, abar, htm)
    else: 
        SolveForward_pb(params, institutions, abar, htm, beta)
    




#########################################################################################
#########################################################################################
# DOWN FROM HERE THE FUNCTIONS AND CLASS ARE USED TO ESTIMATE THE MODEL PARAMETERS 
# USING SIMULATED METHOD OF MOMENTS
#########################################################################################



#########################################################################################
# This function simulates the moments given our model, the parameters and our weighting matrix.  
# So these moments are the 'simulated fake' ones. 

#@njit(cache=True)
def simulate_moments(params, institutions_pre, institutions_post, abar, weights, htm):

    # Simulate Model
    #S_pre, V_emp_pre, V_unemp_pre, c_emp_pre, c_unemp_pre, Vss_emp_pre, Vss_uemp_pre, css_emp_pre, css_uemp_pre, survival_pre, benefits_pre      = \
    #    SolveMultiTypeModel(params,institutions_pre, abar, htm)

    #S_post, V_emp_post, V_unemp_post, c_emp_post, c_unemp_post, Vss_emp_post, Vss_uemp_post, css_emp_post, css_uemp_post, survival_post, benefits_post = \
    #    SolveMultiTypeModel(params,institutions_post, abar, htm)

    cons_pre, S_pre, assets_pre, asset_grid_pre, benefits_pre    = \
        SolveMultiTypeModel(params,institutions_pre, abar, htm)
    
            
    cons_post, S_post, assets_post, asset_grid_post, benefits_post = \
        SolveMultiTypeModel(params,institutions_post, abar, htm)
    # Return Moments
    
    # We exclude the first observed moment to avoid on job, job search. This is alligned with excluding the first moment in 'matching moments' 
    # This means that we don't ever look at the hazard rate in period 1. 
    moments_pre = weights @ S_pre[:,0:35]       
   
    moments_post = weights @ S_post[:,0:35]

    moments_model = np.hstack((moments_pre, moments_post))
    

    return moments_model

#########################################################################################
# Here we load the observed moments from the Hungarian data set given by DellaVigna et al

#@njit(cache=True)
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
    var_pre = sd_pre*2
    var_post = sd_post**2
    var = np.hstack((var_pre,var_post))
    cov = np.eye(len(var))*var

    return target, cov

#########################################################################################
# This function finds the sum of squared errors between the simulated moments and the observed moments. 
# this is done with already given parameter values which we use as a starting point to estimate the parameters. 

#@njit(cache=True)
def sse(params, target, W, institutions_pre, institutions_post, abar, weights, htm):
    simmoments = simulate_moments(params, institutions_pre, institutions_post, abar, weights, htm)

    # Deviations between target moments and simulated moments:
    err= target - simmoments
    # Calculate SSE
    SSEval = err.T @ W @ err

    return SSEval

#########################################################################################
# This class is used to iterate over candidate parameters.  
# We call this with a minimizer to find the parameters that minimize the SSE. 

class smm:
    def __init__(self, params_full, target, W, institutions_pre, institutions_post, abar, weights, htm, disp=False):
        self.iter = 0
        self.params_full = params_full
        self.target = target
        self.W = W # Covariance matrix, the weighting matrix for the GMM estimation.
        self.institutions_pre = institutions_pre
        self.institutions_post = institutions_post
        self.abar = abar
        self.weights = weights  # weighting of asset distribution
        self.disp = disp
        self.L = np.linalg.cholesky(W)
        self.htm = htm

    def sse(self,params):
        # Deviations between target moments and simulated moments:
        self.params_full.update(params)
        params_full_vec = np.array(self.params_full['value'])
        simmoments = simulate_moments(params_full_vec, self.institutions_pre, self.institutions_post, self.abar, self.weights, self.htm)

        # Deviations between target moments and simulated moments:
        err= self.target - simmoments
        # Calculate SSE
        SSEval = err.T @ self.W @ err

        self.iter = self.iter+1
        if self.disp:
            print('Iter: {:.0f}; Current SSE: {:10.3f}'.format(self.iter, sse))

        return SSEval
    
    def criterion(self,params):
        # Deviations between target moments and simulated moments:
        self.params_full.update(params)
        params_full_vec = np.array(self.params_full['value'])
        simmoments = simulate_moments(params_full_vec, self.institutions_pre, self.institutions_post, self.abar, self.weights, self.htm)

        # Deviations between target moments and simulated moments:
        err= self.target - simmoments
        # Calculate SSE
        # SSEval = err.T @ self.W @ err

        #L = np.linalg.cholesky(self.W)
        weighted_residuals = err @ self.L

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

def SolveMultiTypeModel(params,institutions, abar, htm):
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

    if len(params_vec)==6:
        delta, gamma, eta, k1, lmbda, N = params_vec
        kvals = [k1]
        shares = np.array([1.0])

    # Variables for 2-type estimation
    elif len(params_vec)==8:
        delta, gamma, eta,k1, lmbda, N, k2, q1 = params_vec
        kvals = [k1, k2]
        shares = np.array([q1, 1-q1])

    # Variables for 3-type estimation
    elif len(params_vec)==10:
        delta, gamma, eta, k1, lmbda, N, k2, k3, q1, q2 = params_vec
        kvals = [k1, k2, k3]
        shares = np.array([q1, q2, 1-q1-q2] )
        

    else:
        raise ValueError('Number of types not supported. Please use 1, 2 or 3 types.')
        

    # Validity Check
    if np.any(shares < 0):
        raise ValueError("Type shares must be non-negative.")
    
    if not np.isclose(shares.sum(), 1.0):
        raise ValueError("Type shares must sum to 1.")

    # htm is already in institutions
    n_a = int(institutions[0])

    weights = make_weights(
        htm,
        abar,
        n_a
    )
    #------------------------------------------------------------------
    # SOLVE THE MODEL FOR EACH TYPE
    #------------------------------------------------------------------ 
    
    S_types = []
    cons_types = []
    survival_types = []
    assets_types = []
    asset_grid_out = None
    benefits_out = None
    

    for k_j in kvals:
        params_j = np.array([delta, gamma, eta, k_j, lmbda, N]) 
        cons, S, survival, assets, asset_grid, benefits = SolveForward(params_j, institutions, abar, htm)

        cons_types.append(cons)
        S_types.append(S)
        survival_types.append(survival)
        assets_types.append(assets)
        asset_grid_out = asset_grid
        benefits_out = benefits
        

    #------------------------------------------------------------------
    # Aggregate survival across types
    #------------------------------------------------------------------
    survival_agg = np.zeros_like(survival_types[0])

    for j in range(len(kvals)):

        # Weight type j by its population share,
        # but preserve all asset states
        survival_agg += shares[j] * survival_types[j]

   
    #------------------------------------------------------------------
    # Aggregate hazard
    #------------------------------------------------------------------
    s_agg = np.zeros_like(survival_agg)
    Tplus1 = survival_agg.shape[1]

    for t in range(Tplus1 - 1):

        # Which asset states still have meaningful survival mass?
        valid = survival_agg[:, t] > 1e-7

        s_agg[valid, t] = (
            survival_agg[valid, t]
            - survival_agg[valid, t + 1]
        ) / survival_agg[valid, t]


    # Last period: copy previous hazard
    s_agg[:, -1] = s_agg[:, -2]

    cons_agg = np.zeros_like(cons_types[0])

    for t in range(cons_agg.shape[1]):

        denominator = survival_agg[:, t]

        numerator = np.zeros(cons_agg.shape[0])

        for j in range(len(kvals)):
            numerator += (
                shares[j]
                * survival_types[j][:, t]
                * cons_types[j][:, t]
            )

        valid = denominator > 1e-10

        cons_agg[valid, t] = (
            numerator[valid]
            / denominator[valid]
        )
    assets_agg = np.zeros_like(assets_types[0])

    for t in range(assets_agg.shape[1]):

        denominator = survival_agg[:, t]

        numerator = np.zeros(assets_agg.shape[0])

        for j in range(len(kvals)):
            numerator += (
                shares[j]
                * survival_types[j][:, t]
                * assets_types[j][:, t]
            )

        valid = denominator > 1e-10

        assets_agg[valid, t] = (
            numerator[valid]
            / denominator[valid]
        )

    return cons_agg, s_agg, assets_agg, asset_grid_out, benefits_out

def SolveMultiTypeModel2(params,institutions, abar, htm):
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

    if len(params_vec)==6:
        delta, gamma, eta, k1, lmbda, N = params_vec
        kvals = [k1]
        shares = np.array([1.0])

    # Variables for 2-type estimation
    elif len(params_vec)==8:
        delta, gamma, eta,k1, lmbda, N, k2, q1 = params_vec
        kvals = [k1, k2]
        shares = np.array([q1, 1-q1])

    # Variables for 3-type estimation
    elif len(params_vec)==10:
        delta, gamma, eta, k1, lmbda, N, k2, k3, q1, q2 = params_vec
        kvals = [k1, k2, k3]
        shares = np.array([q1, q2, 1-q1-q2] )
        

    else:
        raise ValueError('Number of types not supported. Please use 1, 2 or 3 types.')
        

    # Validity Check
    if np.any(shares < 0):
        raise ValueError("Type shares must be non-negative.")
    
    if not np.isclose(shares.sum(), 1.0):
        raise ValueError("Type shares must sum to 1.")

    # htm is already in institutions
    n_a = int(institutions[0])

    weights = make_weights(
        htm,
        abar,
        n_a
    )
    #------------------------------------------------------------------
    # SOLVE THE MODEL FOR EACH TYPE
    #------------------------------------------------------------------ 
    
    S_types = []
    V_emp_types = []
    V_uemp_types = []
    c_emp_types = []
    c_uemp_types = []

    Vss_emp_types = []
    Vss_uemp_types = []
    css_emp_types = []
    css_uemp_types = []
    survival_types = []

    benefits_out = None
    

    for k_j in kvals:
        params_j = np.array([delta, gamma, eta, k_j, lmbda, N]) 
        S, V_emp, V_uemp, c_emp, c_uemp, Vss_emp, Vss_uemp, css_emp, css_uemp, survival, benefits = SolveModel(params_j, institutions, abar, htm)
        S_types.append(S)
        V_emp_types.append(V_emp)
        V_uemp_types.append(V_uemp)
        c_emp_types.append(c_emp)
        c_uemp_types.append(c_uemp)

        Vss_emp_types.append(Vss_emp)
        Vss_uemp_types.append(Vss_uemp)
        css_emp_types.append(css_emp)
        css_uemp_types.append(css_uemp)

        survival_types.append(survival)

        benefits_out = benefits
        

    #------------------------------------------------------------------
    # Aggregate survival across types
    #------------------------------------------------------------------
    survival_agg = np.zeros_like(survival_types[0])

    for j in range(len(kvals)):

        # Weight type j by its population share,
        # but preserve all asset states
        survival_agg += shares[j] * survival_types[j]

    #------------------------------------------------------------------
    # Aggregate hazard
    #------------------------------------------------------------------
    s_agg = np.zeros_like(survival_agg)
    Tplus1 = survival_agg.shape[1]

    for t in range(Tplus1 - 1):

        # Which asset states still have meaningful survival mass?
        valid = survival_agg[:, t] > 1e-7

        s_agg[valid, t] = (
            survival_agg[valid, t]
            - survival_agg[valid, t + 1]
        ) / survival_agg[valid, t]


    # Last period: copy previous hazard
    s_agg[:, -1] = s_agg[:, -2]

    return s_agg, V_emp_types, V_uemp_types, c_emp_types, c_uemp_types, Vss_emp_types, Vss_uemp_types, css_emp_types, css_uemp_types, survival_agg, benefits_out