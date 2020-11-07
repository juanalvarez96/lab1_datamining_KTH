import pdb
from numpy.random import randint
import numpy as np
class MinHashing():
    def __init__(self, signature_num, M):
        self.signature_num = signature_num
        self.M = M
    
    def prime(self,n):
        np=[]
        isprime=[]
        for i in range (n+1,n+200):
            np.append(i)
        for j in np:
            val_is_prime = True
            for x in range(2,j-1):
                if j % x == 0:
                    val_is_prime = False
                    break
            if val_is_prime:
                isprime.append(j)
        return min(isprime)
    def change(self, indexes, rIndex):
        hashed_values = M1[rIndex, -self.signature_num:]
        for index in indexes:
            row = minHashing[:, index]
            result = np.where(hashed_values < row)
            if len(result)>0:
                for ind in result:
                    minHashing[ind, index] = hashed_values[ind]

    M1 = np.hstack((self.M, permutations))
