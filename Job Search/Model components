########################################################################### 
##################### We import the needed packages #######################
import numpy as np
from time import process_time
import matplotlib.pyplot as plt
import scipy
from scipy import interpolate  # type: ignore # Interpolation routines
from matplotlib.gridspec import GridSpec

########################################################################### 
######################## We determine parameters ##########################

delta = 0.93
gamma = 0.9
eta = 1
k = 300
k1 = 300
k2 = 250
k3 = 200
q1 = 0.3
q2 = 0.3
lmbda = 4.9
N = 6

abar = [0, 10000]
n_a = 100
n_c = 500

T1 = 6
T2 = 18
T3 = 24
T = 44
b1= 222
b2 = 222
b3 = 114 
welfare = 90
w = 675
R = 0.01

params_single = np.array([delta, gamma, eta, k, lmbda, N])
params_multi = np.array([delta, gamma, eta, k1, k2, k3, q1, q2,lmbda, N])
institutions = np.array([ n_a, n_c, T1, T2, T3, T, b1, b2, b3, welfare, w, R])


#########################################################################################
#########################################################################################
# HERE WE MAKE ALL THE FUNCTIONS THAT WE NEED FOR THE CLASSSES 
#########################################################################################

#########################################################################################
# We make the gain-loss function
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
# We make the function for the optimal search effort, given the value of employment and unemployment
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

def employment_BI(Vss, ref_path, delta, eta, lmbda, R, w, abar, T, N, n_a, n_c):
    abar[0] = np.maximum(np.finfo(float).eps, abar[0])              # vælger numerisk nul, så vi kan tage log. 
    a = np.linspace(abar[0], abar[1], n_a).reshape((n_a, 1))        # laver en n_a x 1 matrice med værdierne fra 0 til maks asset level. 

    c = np.empty((n_a, n_c))    # For hver state kan man vælge n_c forskellige niveauer af forbrug i perioden
    for i in range(n_a):        # vi kigger på alle rækkerne og tilføjer forbrugsmuligheder fra 0 og op til maks forbrug som er givet ved assets + løn.  
        c[i, :] = np.linspace(abar[0], (a[i]+w), n_c).reshape((1, n_c))     # Laver det om til en vektor med 1 række og n_c kolonner. 

    a1 = (a - c + w) * (1 + R)   # assets i næste periode. Det er en n_a x n_c matrix. 

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
    
    def income_path_(self):                          
        '''Laver alle de mulige income paths. 
            En for alle tidspunkter hvorpå man finder et job, inklusiv muligheden for aldrig at finde et job
        '''
        benefits = self.benefit_path_()
        income_path = np.tile(benefits,(self.T+1,1))
        for j in np.arange(self.T): 
            income_path[j,j:] = self.w
        return income_path
    
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

    def __init__(self, delta=0.9, eta = 1, lmbda = 4.9, abar=[0, 20], n_a=50, n_c=100, w=15, R = 0.05 ):
        """Initializer"""
        self.delta = delta    # Discount factor
        self.eta = eta      # gain-loss value parameter
        self.lmbda = lmbda  # loss aversion parameter
        self.abar = abar    # min and max savings
        self.n_a = n_a      # grid size for state variable
        self.n_c = n_c      # grid size for choice grid
        self.w = w          # wage
        self.R = R          # interest rate

        self.abar[0] = np.maximum(np.finfo(float).eps, self.abar[0])       # vælger den højeste værdi af minimumsværdien for abar og 'numerisk 0'. Vi vil ikke under numerisk 0.  
        # truncate lower bound at smallest positive float number
        # parameter dependent varibles
        self.a = np.linspace(self.abar[0], self.abar[1], n_a).reshape((n_a, 1))  # vi laver en vektor med state space fra 0 til abar og med det antal grid punkter vi vil have, n_a.  
        self.c = np.empty((n_a, n_c))  # starter consumption matrix med rækker for alle states og så et finere grid i kolonnerne for mulige consumption valg. 
        
        for i in range(n_a):            # fordi rækkerne er state-grids, så derfor skal vi have en række for hver state vi vil kigge på
            self.c[i, :] = np.linspace(self.abar[0], self.a[i]+self.w, n_c).reshape((1, n_c))  # vi laver hver række så den går fra '0' til den værdi i state grid, som vi kigger på. Altså, det vil svare til at vi kigger på alle mulige værdier for V(A=10)
                                                                                        # consumption choice bliver så lidt finere (det er kolonnerne her) og tager n_c grid points. 
                                                                                        # så hver række repræsenterer alle consumption choices givet en state value. 

           #Bellman operator, V0 is one-dim vector of values on state grid                                 
    def bellman(self, V0, R):           
        interp = interpolate.interp1d(self.a[:, 0], V0, bounds_error=False, fill_value="extrapolate")
        V = 0
        a1 = (self.a - self.c + self.w)*(1+R)       # vi laver a1 som funktion af hvad du forbruger     'a' er en 1 x n_a vektor
        V = interp(a1)                              # a1 bliver en n_a x n_c matrix     V svarer til de 100 forskellige niveauer af nytte i næste periode
                                                    # a1 er en funktion af c (de andre er faste), så får vi en værdi som ligger mellem grid points ved at interpolere.
                                                    # der er ikke noget med nytte. Er det med vilje fordi den endelige værdi af V altid være den som maksimerer nytten?
        matV1 = u(self.c, self.w, self.eta, self.lmbda ) + self.delta * V      # c er en n_a x n_c vektor   a er en na x 1 vektor a1 bliver så en n_a x n_c matrix. Det gør V så også 
                                                    # så her lægges en n_a x n_c matrix til en n_a x n_c matrix, hvilket giver en n_a x n_c matrix.
        i_max = np.argmax(matV1, axis=1)            # vi kigger på den kolonne som giver den højeste nytte for hver række. n_a x 1 matrix
        # (column) index of optimal choices
        V1 = matV1[np.arange(self.n_a), i_max]          # Her går den igennem alle n_a rækker, og finder den kolonne i hver række, som giver den højeste værdi. 
        c1 = self.c[np.arange(self.n_a), i_max]         # den her viser, hvad det optimale forbrug så viste sig at være. 

        return V1, c1


def vfi(self, maxiter=1000, tol=1e-15, callback=None):
    """Solves the model using VFI (successive approximations)"""
    tic = process_time()  # Start the stopwatch / counter

    V0 = np.log(self.a[:, 0])  # on first iteration assume consuming everything
    
    for iter in range(maxiter):
        V1, c1 = self.bellman(V0, self.R)
        if callback:
            callback(iter, self.a, V1, c1)  # callback for making plots
        if np.all(abs(V1 - V0) < tol):
            toc = process_time()  # Stop the stopwatch / counter
            print("Optimal consumption of assets solved in", iter, "iterations, using", round(toc - tic, 5), "seconds")
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

def SolveModel(delta, gamma, eta, k, lmbda, abar, n_a, n_c, T1, T2, T3, T, N, b1, b2, b3, welfare, w, R ):
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
    #Steady State Values for employment and unemployment
    emp = ss_value(delta, abar.copy(), n_a, n_c, w, R, w, eta, lmbda)
    Vss_emp, css_emp = emp.solve()
    uemp = ss_value(delta, abar.copy(), n_a, n_c, welfare, R, welfare, eta, lmbda)
    Vss_uemp, css_uemp = uemp.solve()

    # Income paths
    model = ref_inc_path(w, b1, b2, b3, welfare, T1, T2, T3, T, N, eta )
    ref_path_long = model.ref_path_long_()
    benefits = model.benefit_path_()

    #Find full value of employment and unemployment for all asset levels and all periods
    V_emp, c_emp = employment_BI(Vss_emp , ref_path_long, delta, eta, lmbda, R, w, abar, T, N, n_a, n_c)

    
    # Asset grid
    abar[0] = np.maximum(np.finfo(float).eps, abar[0])              # vælger numerisk nul, så vi kan tage log. 
    a = np.linspace(abar[0], abar[1], n_a).reshape((n_a, 1))        # laver en n_a x 1 matrice med værdierne fra 0 til maks asset level. 

    # consumption possibilites
    #c = np.empty((n_a, n_c))    # For hver state kan man vælge n_c forskellige niveauer af forbrug i perioden
    #for i in range(n_a):        # vi kigger på alle rækkerne og tilføjer forbrugsmuligheder fra 0 og op til maks forbrug som er givet ved assets + løn.  
    #    c[i, :] = np.linspace(abar[0], (a[i]+w), n_c).reshape((1, n_c))     # Laver det om til en vektor med 1 række og n_c kolonner. 
    
    #a1 = (a - c + w) * (1 + R)   # assets i næste periode
    
    # Solve for optimal search effort and value functions for unemployment
    V_uemp = np.zeros((n_a, T+1))   # placeholder variable for getting the job in period in period T with asset value a. 
    c_uemp = np.zeros((n_a, T+1))   # associated optimal consumption, first period after job start, not really that relevant though...
    S = np.zeros((n_a, T+1))   # placeholder variable for optimal search effort in period t with asset value a.

    # Terminal Condition
    V_uemp[:, T] = Vss_uemp # Once we reach T periods, the value of being uemployed is equal to the steady state value of being unemployed.
    c_uemp[:, T] = css_uemp # Once we reach T periods, the optimal consumption is equal to the steady state consumption of being unemployed.
    S[:, T] = 0 # One we reach T searching is no longer possible


    # Backwards induction to solve for optimal search effort and value functions for unemployment
    for t in np.arange(T-1, -1, -1):
         # Current unemployment income
         y = benefits[t]

         # consumption possibilites
         c = np.empty((n_a, n_c))    # For hver state kan man vælge n_c forskellige niveauer af forbrug i perioden
         for i in range(n_a):        # vi kigger på alle rækkerne og tilføjer forbrugsmuligheder fra 0 og op til maks forbrug som er givet ved assets + løn.  
             c[i, :] = np.linspace(abar[0], (a[i,0]+y), n_c) #.reshape((1, n_c))     # Laver det om til en vektor med 1 række og n_c kolonner.

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

         V_uemp[:, t] = matV[np.arange(n_a), i_max]               # her opdateres værdierne givet det optimale forbrugsval. Det bliver så genbrugt til næste periode.
         c_uemp[:, t] = c[np.arange(n_a), i_max]
         S[:, t] = S_choices[np.arange(n_a), i_max]

    # Calculate Survival Function
    survival = np.ones((n_a,T+1))
    for t in np.arange(1,T+1):
         survival[:,t] = survival[:,t-1] * (1-S[:,t-1])

    return S, V_emp, V_uemp, c_emp, c_uemp, Vss_emp, Vss_uemp, css_emp, css_uemp, survival, benefits
                