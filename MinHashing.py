from numpy.random import randint
import numpy as np
import random
class minHashing():

    def __init__(self, signature_num, M):
        self.signature_num = signature_num
        self.M = M
        self.a_coef = []
        self.b_coef = []
        self.minHashing
    
    def prime1(self, n):
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
    
    def generate_coefs(self):
        values = random.sample(range(1, len(self.M)), self.signature_num*2)
        self.a_coef = values[0:self.signature_num]
        self.b_coef = values[self.signature_num:(self.signature_num*2)]
    
    def generate_perms(self):
        perms =  np.empty((self.signature_num, 0)).tolist()
        k = 0
        self.generate_coefs() # Generate coefficients
        prime = self.prime1(len(self.M)) # Get next prime to M number of rows
        for perm in perms:
            #pdb.set_trace()
            i = 0
            while i<len(self.M):
                val = (self.a_coef[k]*i + self.b_coef[k])%prime
                perm.append(val)
                i = i+1
            k = k+1
        permutations = np.array(perms).T
        M1 = np.hstack((self.M, permutations))
        return M1
    
    def change(self, M1, minHashing, indexes, rIndex):
        hashed_values = M1[rIndex, -self.signature_num:]
        for index in indexes:
            row = minHashing[:, index]
            result = np.where(hashed_values < row)
            if len(result)>0:
                for ind in result:
                    minHashing[ind, index] = hashed_values[ind]

    
    def minHashing(self):
        M1 = self.generate_perms()
        self.minHashing = np.full([self.M.shape[1],self.signature_num], 999999, dtype=np.int64).T
        for rIndex in range(0, M1.shape[0]-1):
            indexes = np.where(self.M[rIndex,:]==1)[0]
            if len(indexes != 0):
                self.change(M1, indexes, rIndex)
        



    

    
