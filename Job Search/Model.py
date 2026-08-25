### Find the Steady State Value of the Bellman Equation
class ss_value:
    """Class to implement the stochastic cake eating model with discretized choice"""

    def __init__(self, delta, abar, n_a, n_c, w, R ):
        """Initializer"""
        self.delta = delta    # Discount factor
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
        """" Det her er nok forkert. Vi skal have tilføjet + w i forbrugsmuligheden"""

        """Bellman operator, V0 is one-dim vector of values on state grid"""
    def bellman(self, V0, R):
        interp = interpolate.interp1d(self.a[:, 0], V0, bounds_error=False, fill_value="extrapolate")
        V = 0
        a1 = (self.a - self.c + self.w)*(1+R)       # vi laver a1 som funktion af hvad du forbruger     'a' er en 1 x n_a vektor
        V = interp(a1)                              # a1 bliver en n_a x n_c matrix     V svarer til de 100 forskellige niveauer af nytte i næste periode
                                                    # a1 er en funktion af c (de andre er faste), så får vi en værdi som ligger mellem grid points ved at interpolere.
                                                    # der er ikke noget med nytte. Er det med vilje fordi den endelige værdi af V altid være den som maksimerer nytten?
        matV1 = np.log(self.c) + self.delta * V      # c er en n_a x n_c vektor   a er en na x 1 vektor a1 bliver så en n_a x n_c matrix. Det gør V så også 
                                                    # så her lægges en n_a x n_c matrix til en n_a x n_c matrix, hvilket giver en n_a x n_c matrix.
        i_max = np.argmax(matV1, axis=1)            # vi kigger på den kolonne som giver den højeste nytte for hver række. n_a x 1 matrix
        # (column) index of optimal choices
        V1 = matV1[np.arange(self.n_a), i_max]          # Her går den igennem alle n_a rækker, og finder den kolonne i hver række, som giver den højeste værdi. 
        c1 = self.c[np.arange(self.n_a), i_max]         # den her viser, hvad det optimale forbrug så viste sig at være. 

        return V1, c1
    

### Solve by Value Function Iteration
def vfi(self, maxiter=1000, tol=1e-10, callback=None):
    """Solves the model using VFI (successive approximations)"""
    tic = process_time()  # Start the stopwatch / counter

    V0 = np.log(self.a[:, 0])  # on first iteration assume consuming everything
    """eventuelt tilføj w i den initialle V0"""
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





# Generate income path and reference path
class ref_inc_path():
    def __init__(self,w , b1, b2, b3, welfare, T1, T2, T3, T, N, eta ):
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
        self.n_periods = self.T + 1 # Perioder i alt (job kan starte i periode 0 til T, og så er der en ekstra periode hvor man aldrig finder et job)
        self.never_emp = self.T + 1 # Periode hvor man aldrig finder et job. Vi laver en ekstra række i income path, som repræsenterer dette.
        

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
        income_path = np.tile(benefits,(self.T+2,1))
        for j in np.arange(self.n_periods): 
            income_path[j,j:] = self.w

        income_path[self.never_emp, :] = self.welfare
        return income_path
    
    def ref_path_(self):             # the reference point is given as the arithmetic average of the income in the 5 periods leading up to the current one. 
        if self.eta > 0: 
            ref_path = np.zeros((self.T+2, self.T))
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
            
        add = np.zeros( (self.T+2, self.N)  )

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
        # Række til dem der aldrig får et job
        ref_path_long[self.never_emp, self.T:] = self.welfare
        
        return ref_path_long


