import pyspark
import pdb
sc = pyspark.SparkContext("local[*]", "lab1")
from pyspark.sql import SparkSession

from Shingling import Shingling
from CompareSets import CompareSets

def cleanDoc(doc):
    doc = doc.lower()
    filter_chars = ['.', '-', '´', ',', '(', ')', '‘', '’']
    filtered_doc = ''.join((filter(lambda x: x not in filter_chars, doc)))
    return filtered_doc
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

#Select n documents from df_final
docs = df_final.sort(df_final.content.desc()).limit(n).collect()

# Instantiate Shingling class
shingler = Shingling(k=k)
shinglings = list()
hashes = list()
# Get shingles and store
for doc in docs:
    #pdb.set_trace()
    doc = doc.content # Now we have a string
    doc = cleanDoc(doc)
    shingler.make_shingles(doc=doc)
    shinglings.append(shingler.shinglings)
    hashes.append(shingler.shinglings_hash)
    shingler.clean()
    #print(shingler.shinglings_hash)

# Compute similarity between two
comparator = CompareSets()
for hashi in hashes:
    comparator.jaccard_sim(hashes[0], hashi)




