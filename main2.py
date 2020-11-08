import pdb
import numpy as np
from Shingling import Shingling
from CompareSets import CompareSets
from CompareSignature import CompareSignature
import random

def cleanDoc(doc):
    doc = doc.lower()
    filter_chars = ['.', '-', '´', ',', '(', ')', '‘', '’']
    filtered_doc = ''.join((filter(lambda x: x not in filter_chars, doc)))
    return filtered_doc

docs = ["The sky is blue and the sun is bright.",
        "The sun in the sky is bright.",
        "We can see sun is bright, the sky is blue." ]


# Size of shingles
k = 3

# Instantiate Shingling class
shingler = Shingling(k=k)
shinglings = list()
hashes = list()
for doc in docs:
    #pdb.set_trace()
    doc = cleanDoc(doc)
    shingler.make_shingles(doc=doc)
    shinglings.append(shingler.shinglings)
    hashes.append(set(shingler.shinglings_hash))
    shingler.clean()
    #print(shingler.shinglings_hash)

# Compute similarity between two
comparator = CompareSets()
for hashi in hashes:
    comparator.jaccard_sim(hashes[0], hashi)

# Transform hashes back to lists
hashes_transformed = []
for item in hashes: hashes_transformed.append(list(item))

# Generate characteristic matrix
merged = sum(hashes_transformed, []) # flatten list
merged_unique = set(merged) # Get uniques
#pdb.set_trace()
M = np.zeros([len(merged_unique), len(docs)])
merged = list(merged_unique)

for i in range(0, len(docs)):
    big = np.array(merged)
    small = np.array(hashes_transformed[i])
    zero = np.zeros(len(M[:, i]))
    indices = np.in1d(big, small).nonzero()[0]
    zero[indices]=1
    M[:,i]=zero

print("Shape of characteristic matrix: {}".format(M.shape))

# Generate permutated rows
from numpy.random import randint
#pdb.set_trace()
signature_num = 6
prime = 15
values = random.sample(range(1, len(M)), signature_num*2)
a_coef = values[0:signature_num]
b_coef = values[signature_num:(signature_num*2)]

#for i in range ()
perms =  np.empty((signature_num, 0)).tolist()
k = 0
for perm in perms:
    #pdb.set_trace()
    i = 0
    while i<len(M):
        val = (a_coef[k]*i + b_coef[k])%prime
        if val not in perm: 
            perm.append(val)
            i = i+1
        else:
            #print("Repeated value:{}\nArray:{}\n".format(val, perm))
            #print("Searching for new unique value...\n")
            a_coef[k]=np.random.randint(0, len(M))
            b_coef[k]=np.random.randint(0, len(M))
    k = k+1
permutations = np.array(perms).T

# Merge M with permutated rows
M1 = np.hstack((M, permutations))

# Perform min hashing
minHashing = np.full([len(hashes),signature_num], 999999, dtype=np.int64).T

#pdb.set_trace()

def change(indexes, rIndex):
    hashed_values = M1[rIndex, -signature_num:]
    for index in indexes:
        row = minHashing[:, index]
        result = np.where(hashed_values < row)
        if len(result)>0:
            for ind in result:
                minHashing[ind, index] = hashed_values[ind]



for rIndex in range(0, M1.shape[0]-1):
    indexes = np.where(M[rIndex,:]==1)[0]
    if len(indexes != 0):
        change(indexes, rIndex)



# Compare signatures
sig_comparator = CompareSignature()
#sig_comparator.comparator(minHashing[:, 0], minHashing[:, 2])
#pdb.set_trace()

# LSH
b = 3 # Number of bands
r = 2 # Number of rows per band
k = 60 # Number of buckets
t = (1/b)**(1/r) # Threshold
aux = 0
import mmh3
results=[]
# hash to bucket function:
def hash_to_bucket(e, B):
    i = mmh3.hash128(str(e))
    p = i / float(2**128)
    for j in range(0, B):
        if j/float(B) <= p and (j+1)/float(B) > p:
            return j+1
    return B
aux2 = 0
#pdb.set_trace()
while aux < len(minHashing)/r: 
    aux = aux + 1
    LSHs = {}
    for docId in range(0, len(docs)):
        vector = minHashing[aux2:aux2+r, docId]
        bucket = hash_to_bucket(vector, k)
        if bucket in LSHs:
            LSHs[bucket].append(docId)
        else:
            LSHs[bucket] = [docId]
        results.append(LSHs)
    aux2=aux2+r

print(results)
pdb.set_trace()
    




