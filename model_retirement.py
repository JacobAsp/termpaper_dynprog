# Zurcher class: Contains model parts for Rust's engine repplacement model Rust(Ecta, 1987)

#import packages
import numpy as np
import time
import pandas as pd
import statsmodels.api as sm

class retirement():
    def __init__(self,**kwargs):
        self.setup(**kwargs)

    def setup(self,**kwargs):     
  
        # a) parameters
        # Spaces
        #self.n = 175                      # Number of grid points
        #self.max = 108                    # Max age  

        # structual parameters

        self.p = np.array([0.0937, 0.4475, 0.4459, 0.0127])   # Transition probability
        self.alpha = 11.7257                                     # consumption preference
        self.phi = 2.45569                                      # leisure time preference
        self.beta = 0.97                                    # Discount factor

        ages = np.arange(50, 103)           # Age 50 to 102
        married_states = [0,1]              # Married or not married
        retired_states = [0,1]              # Retired or not retired   
        atp_grid = np.linspace(0, 6.5, 65)  # ATP points grid

        def wage(age): 
            return 300 + 5 * age  # placeholder
        
        def pension(age_ret, atp, married): 
            # Implement ATP + basic pension formula
            ba = 38600  # base amount
            bp = 0.96 * ba if not married else 0.785 * ba
            supplement = max(0, 0.555 * ba - 0.6 * atp * ba)
            supplement = min(supplement, 0.555 * ba)
            atp_pension = 0.6 * atp * ba
            adj =  (1.007**(12*(age_ret - 65)) if age_ret > 65 else 0.995**(12*(65 - age_ret)))
            return adj * (bp + supplement + atp_pension)

        # Leisure
        def leisure(retired): 
            return 1.0 if retired else 0.55
        
        # b. update baseline parameters using keywords
        for key,val in kwargs.items():
            setattr(self,key,val) 

        # c. Create grid
        self.create_grid()

    def create_grid(self):
        self.grid = np.arange(0,self.n) # milage grid
        self.cost = 0.001*self.c*self.grid  # cost function
        self.state_transition() 

    def state_transition(self):
        '''Compute transition probability matrixes conditional on choice'''
        p = np.append(self.p,1-np.sum(self.p)) # Get transition probabilities
        P1 = np.zeros((self.n,self.n)) # Initialize transition matrix
        # Loop over rows
        for i in range(self.n):
            # Check if p vector fits entirely
            if i <= self.n-len(p):
                P1[i][i:i+len(p)]=p
            else:
                P1[i][i:] = p[:self.n-len(p)-i]
                P1[i][-1] = 1.0-P1[i][:-1].sum()

        # conditional on d=1, replacement
        P2 = np.zeros((self.n,self.n))
        # Loop over rows
        for i in range(self.n):
            P2[i][:len(p)]=p
        self.P1 = P1
        self.P2 = P2

    def utility(c, f):
        return alpha * np.log(c + 1e-6) + phi * np.log(f)
    

    def bellman(V, survival, max_age=100):
        """Performs one value iteration step"""
        V_new = np.copy(V)

        for a in range(50, max_age):      # age
            for r in retired_states:      # retired
                for m in married_states:  # marital
                    for i, atp in enumerate(atp_grid):
                        key = (a, r, m, i)

                        if r == 1:
                            # Retired: deterministic future
                            b = pension(a, atp, m)
                            c = b
                            f = leisure(1)
                            u = utility(c, f)
                            cont_val = 0 if a == max_age else beta * survival[a] * V[a+1, 1, m, i]
                            V_new[key] = u + cont_val
                        else:
                            # Choice: work (0) or retire (1)
                            # Work
                            w = wage(a)
                            atp_new = update_atp(atp, a, gamma, sigma2)
                            i_new = np.argmin(np.abs(atp_grid - atp_new))
                            c_w = w
                            u_w = utility(c_w, leisure(0))
                            cont_w = 0 if a == max_age else beta * survival[a] * V[a+1, 0, m, i_new]

                            # Retire
                            b = pension(a, atp, m)
                            c_r = b
                            u_r = utility(c_r, leisure(1))
                            cont_r = 0 if a == max_age else beta * survival[a] * V[a+1, 1, m, i]

                            V_new[key] = max(u_w + cont_w, u_r + cont_r)

        return V_new
    
    def update_atp(apt_t, age, gamma, sigma2, epsilon=1e-4):
        log_apt_t = np.log(apt_t + epsilon)
        log_apt_t1 = (
            gamma['const'] +
            gamma['log_apt_t'] * log_apt_t +
            gamma['age'] * age +
            gamma['age_squared'] * age**2 +
            0.5 * sigma2
        )
        return min(np.exp(log_apt_t1), 6.5)  # Cap at statutory max
    
    def bellman(self,ev0,output=1):
        '''Evaluate Bellman operator, choice probability and Frechet derivative - written in integrated value form'''

        # Value of options:
        value_work = -self.cost + self.beta * self.P1 @ ev0 # nx1 matrix
        value_retire = -self.RC - self.cost[0] + self.beta * self.P2 @ ev0   # 1x1

        # recenter Bellman by subtracting max(VK, VR)
        maxV = np.maximum(value_work, value_retire) 
        logsum = (maxV + np.log(np.exp(value_work-maxV)  +  np.exp(value_retire-maxV)))  # Compute logsum to handle expectation over unobserved states
        ev1 = logsum # Bellman operator as integrated value

        if output == 1:
            return ev1

        # Compute choice probability of keep
        pk = 1/(1+np.exp(value_retire-value_work))       
        
        if output == 2:
            return ev1, pk

        # Compute derivative of Bellman operator
        dev1 = self.dbellman(pk)

        return ev1, pk, dev1

    def dbellman(self,pk): 
        '''Compute derivative of Bellman operator'''
        dev1 = np.zeros((self.n,self.n))
        for d in range(2): # Loop over choices 
            if d == 0:
                P = self.P1
                choice_prob =  pk
            else:
                P = self.P2
                choice_prob = 1-pk

            dev1 += self.beta * choice_prob.reshape(-1, 1) * P 
        
        return dev1

    def read_spardata(self, bustypes = [1,2,3,4]): 
        data = np.loadtxt(open("KPS Data/cleaned_spardata"), delimiter=" ", skiprows=1) #datapath
        idx = data[:,0]             # bus id
        bustype = data[:,1]         # bus type
        dl = data[:,4]              # laggend replacement dummy
        d = np.append(dl[1:], 0)    # replacement dummy
        x = data[:,6]               # Odometer

        # Discretize odometer data into 1,2,...,n
        x = np.ceil(x*self.n/(self.max*1000))

        # Montly mileage
        dx1 = x-np.append(0,x[0:-1])
        dx1 = dx1*(1-dl)+x*dl
        dx1 = np.where(dx1>len(self.p),len(self.p),dx1) # We limit the number of steps in mileage

        # change type to integrer
        x = x.astype(int)
        dx1 = dx1.astype(int)

        # Collect in a dataframe
        remove_first_row_index=idx-np.append(0,idx[:-1])
        data = {'id': idx,'bustype':bustype, 'd': d, 'x': x, 'dx1': dx1, 'boolean': remove_first_row_index}
        df= pd.DataFrame(data) 

        # Remove observations with missing lagged mileage
        df = df.drop(df[df.boolean!=0].index)

        # Select bustypes 
        for j in [1,2,3,4]:
            if j not in bustypes:
                df = df.drop(df[df.bustype==j].index) 

        # save data
        dta = df.drop(['id','bustype','boolean'],axis=1)
        
        return dta

    def read_busdata(self, bustypes = [1,2,3,4]): 
        data = np.loadtxt(open("busdata1234.csv"), delimiter=",")
        idx = data[:,0]             # bus id
        bustype = data[:,1]         # bus type
        dl = data[:,4]              # laggend replacement dummy
        d = np.append(dl[1:], 0)    # replacement dummy
        x = data[:,6]               # Odometer

        # Discretize odometer data into 1,2,...,n
        x = np.ceil(x*self.n/(self.max*1000))

        # Montly mileage
        dx1 = x-np.append(0,x[0:-1])
        dx1 = dx1*(1-dl)+x*dl
        dx1 = np.where(dx1>len(self.p),len(self.p),dx1) # We limit the number of steps in mileage

        # change type to integrer
        x = x.astype(int)
        dx1 = dx1.astype(int)

        # Collect in a dataframe
        remove_first_row_index=idx-np.append(0,idx[:-1])
        data = {'id': idx,'bustype':bustype, 'd': d, 'x': x, 'dx1': dx1, 'boolean': remove_first_row_index}
        df= pd.DataFrame(data) 

        # Remove observations with missing lagged mileage
        df = df.drop(df[df.boolean!=0].index)

        # Select bustypes 
        for j in [1,2,3,4]:
            if j not in bustypes:
                df = df.drop(df[df.bustype==j].index) 

        # save data
        dta = df.drop(['id','bustype','boolean'],axis=1)
        
        return dta

    def sim_data(self,N,T,pk): 

        np.random.seed(2020)
        
        # Index 
        idx = np.tile(np.arange(1,N+1),(T,1))  
        t = np.tile(np.arange(1,T+1),(N,1)).T
        
        # Draw random numbers
        u_init = np.random.randint(self.n,size=(1,N)) # initial condition
        u_dx = np.random.rand(T,N) # mileage
        u_d = np.random.rand(T,N) # choice
        
        # Find states and choices
        csum_p = np.cumsum(self.p)
        dx1 = 0
        for val in csum_p:
            dx1 += u_dx>val
        
        x = np.zeros((T,N),dtype=int)
        x1 =  np.zeros((T,N),dtype=int)
        d = np.nan + np.zeros((T,N))
        x[0,:] = u_init # initial condition
        for it in range(T):
            d[it,:] = u_d[it,:]<1-pk[x[it,:]]  # replace = 1 , keep = 0   
            x1[it,:] = np.minimum(x[it,:]*(1-d[it,:]) + dx1[it,:] , self.n-1) # State transition, minimum to avoid exceeding the maximum mileage
            if it < T-1:
                x[it+1,:] = x1[it,:]
                
        
        # reshape 
        idx =  np.reshape(idx,T*N,order='F')
        t = np.reshape(t,T*N,order='F')
        d = np.reshape(d,T*N,order='F')
        x = np.reshape(x,T*N,order='F') + 1 # add 1 to make index start at 1 as in data - 1,2,...,n
        x1 = np.reshape(x1,T*N,order='F') + 1 # add 1 to make index start at 1 as in data - 1,2,...,n
        dx1 = np.reshape(dx1,T*N,order='F')


        data = {'id': idx,'t': t, 'd': d, 'x': x, 'x1': x1, 'dx1': dx1}
        df= pd.DataFrame(data) 

        return df

    def eqb(self, pk):
        # Inputs
        # pk: choice probability

        # Outputs    
        # pp: Pr{x} (Equilibrium distribution of mileage)
        # pp_K: Pr{x.i=Keep}
        # pp_R: Pr{x,i=Replace}
        pl = self.P1 * pk[:,None] + self.P2 * (1 - pk[:,None])

        pp = self.ergodic(pl)

        # joint probabilities
        # a. joint probability of x and i=keep
        pp_K = pk * pp
        # b. joint probability of x and i=replace
        pp_R = (1 - pk) * pp

        return pp, pp_K, pp_R

    def ergodic(self,p):
        #ergodic.m: finds the invariant distribution for an NxN Markov transition probability: q = qH , you can also use Succesive approximation
        n = p.shape[0]
        if n != p.shape[1]:
            print('Error: p must be a square matrix')
            ed = np.nan
        else:
            ap = np.identity(n)-p.T
            ap = np.concatenate((ap, np.ones((1,n))))
            ap = np.concatenate((ap, np.ones((n+1,1))),axis=1)

            # find the number of linearly independent columns
            temp, _ = np.linalg.eig(ap)
            temp = ap[temp==0,:]
            rank = temp.shape[1]
            if rank < n+1:
                print('Error: transition matrix p is not ergodic')
                ed = np.nan
            else:
                ed = np.ones((n+1,1))
                ed[n] *=2
                ed = np.linalg.inv(ap)@ed
                ed = ed[:-1]
                ed = np.ravel(ed)

        return ed