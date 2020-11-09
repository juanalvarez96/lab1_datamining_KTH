import pyspark
import pdb
sc = pyspark.SparkContext("local[*]", "lab1")
from pyspark.sql import SparkSession
from Shingling import Shingling
from CompareSets import CompareSets
import numpy as np
from Shingling import Shingling
from CompareSets import CompareSets
from CompareSignature import CompareSignature
import random

# Number of documents we want
n = 15

# Size of shingles
k = 3

# Read dataset
spark = SparkSession.builder.master('local[*]').appName("lab1").getOrCreate()
sc = spark.sparkContext
df = spark.read.option("header", True).csv('data/airline.csv')

# We are only interested in one columns, the content
df_final = df.select("content")

# Function to get next prime number given a number
def prime1(n):
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
#Select n documents from df_final
docs = df_final.sort(df_final.content.desc()).limit(n).collect()


# Function to clean a single document
def cleanDoc(doc):
    #pdb.set_trace()
    doc = doc.content
    doc = doc.lower()
    filter_chars = ['.', '-', '´', ',', '(', ')', '‘', '’']
    filtered_doc = ''.join((filter(lambda x: x not in filter_chars, doc)))
    return filtered_doc

# Instantiate Shingling class
shingler = Shingling(k=k)
shinglings = list()
hashes = list()
# To load all the shinglings and hashes from each document
# in the shinglings and hashes list
for doc in docs:
    #pdb.set_trace()
    doc = cleanDoc(doc)
    shingler.make_shingles(doc=doc)
    shinglings.append(shingler.shinglings)
    hashes.append(set(shingler.shinglings_hash))
    shingler.clean()
    #print(shingler.shinglings_hash)

# Compute similarity between two
# Instantiate comparator for comparing two sets
comparator = CompareSets()

# Here we compare the doc number 0 with the rest of the docs
# We do this using the CompareSets object.
# Change 0 to select the doc wanted
for hashi in hashes:
    comparator.jaccard_sim(hashes[0], hashi)

# Transform hashes back to lists
hashes_transformed = []
for item in hashes: hashes_transformed.append(list(item))

# Generate characteristic matrix
merged = sum(hashes_transformed, []) # flatten all docs into one list
merged_unique = set(merged) # Get unique shingles from all the documents as a set
#pdb.set_trace()
M = np.zeros([len(merged_unique), len(docs)]) # Define characteristic matrix
merged = list(merged_unique) # Convert flattened set to list

# This loop fills the above-created characteristic matrix with 1s in
# each hash that appears in multiple docs
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
# Signature number are the number of permutations to be performed
signature_num = 50
prime = 0
values = random.sample(range(1, len(M)), signature_num*2)
# Generate coefficients for each hashing function to get a new permuted column
a_coef = values[0:signature_num]
b_coef = values[signature_num:(signature_num*2)]

# Generate the permutations using the hashing function from above
perms =  np.empty((signature_num, 0)).tolist()
k = 0
prime = prime1(len(M)) # Get next prime to M number of rows
for perm in perms:
    #pdb.set_trace()
    i = 0
    while i<len(M):
        val = (a_coef[k]*i + b_coef[k])%prime
        perm.append(val)
        i = i+1
    k = k+1
permutations = np.array(perms).T

# Merge M with permutated rows
M1 = np.hstack((M, permutations))

# Perform min hashing

# Generate the minHashing matrix with very high values by default
minHashing = np.full([len(hashes),signature_num], 999999, dtype=np.int64).T


def change(indexes, rIndex):
    hashed_values = M1[rIndex, -signature_num:]
    for index in indexes:
        row = minHashing[:, index]
        result = np.where(hashed_values < row)
        if len(result)>0:
            for ind in result:
                minHashing[ind, index] = hashed_values[ind]


# Update values in minHashing to get the signatures
for rIndex in range(0, M1.shape[0]-1):
    indexes = np.where(M[rIndex,:]==1)[0]
    if len(indexes != 0):
        change(indexes, rIndex)



# Instantiate signature compartor
sig_comparator = CompareSignature()
# UNCOMMENT FOLLOWING LINE TO GET THE SIMILARITY BETWEEN DOCS 0 AND 2
# YOU CAN ALSO CHANGE THE INDEX TO SELECT ANY DOC YOU WANT.
# sig_comparator.comparator(minHashing[:, 0], minHashing[:, 2]))


# LSH
# The values of b and r are defined to get a thersholf of 0.8!
b = 6 # Number of bands
r = 8 # Number of rows per band
k = 100 # Number of buckets
t = (1/b)**(1/r) # Threshold ~0.8
import mmh3 # Needed to do the hashing for LSH
aux = 0
results=[]
# Function that hashes vector to a bucket, given a number of buckets
def hash_to_bucket(e, B):
    i = mmh3.hash128(str(e))
    p = i / float(2**128)
    for j in range(0, B):
        if j/float(B) <= p and (j+1)/float(B) > p:
            return j+1
    return B

aux2 = 0
# Perform LSH
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

for result in results:
    print(result.values())

pdb.set_trace()


