# -*- coding: utf-8 -*-


import abc
import math
from scipy.linalg import solve

class FitExp(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, N, dt):
        '''
        Initialize exponential fit.
        
        '''
        self.N = N
        self.dt = dt
        
        self.init()
        
    @abc.abstractmethod
    def fitExp(self, data):
        return []
        
    @abc.abstractmethod
    def init(self):
        return None
        
class SLR(FitExp):
    
    def init(self):
        Nm1 = self.N-1
        self.St = Nm1*self.N/2
        self.Stt = Nm1*self.N*(2*self.Nm1+1)/6
        
    def fitExp(self,data):
        
        B = [0,0,0]
        SI = 0
        SII = 0
        SIt = 0
        
        def sum_int(data):
            # Generator function to return a running sum
            i = 0
            b = 0
            while True:
                b += data[i]
                yield b
                i+=1
                
        for i,d in data:
            B[0] = sum_int(data)
            SI+=B[0]
            B[1]+=B[1]*d
            SII +=B**2
            B[2]+=d*i
            SIt += B[0]*i
        
        A = [[self.N, SI, self.St],[SI,SII,SIt],[self.St,SIt,self.Stt]]
        
        X = solve(A,B,sym_pos=True)
        
        sol = {"tau":0,"A":0,"B":0}
        
        sol["tau"] = self.dt/math.log(1-X[1])
        sol["B"] = -X[2]/X[1]
        sol["A"] = math.exp(-1/sol["tau"]*self.dt)*X[0]-sol["B"]
        return sol
                

class crd(object):
    def __init__(self, Rl, exp_obj, dt, N):
        self.exp = exp_obj(N,dt)
        self.optC = Rl/2.99792E+8
        
    def getExt(self, tau, tau0):
        return self.optC*(1/tau-1/tau0)
        
    def allanVar(self):
        return None
        
        
        
        
