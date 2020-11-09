# Itemset and LSH


This is a project from KTH university Data Mining course. The objectives of the project were  to implement the stages of finding textually similar documents based on Jaccard similarity using the shingling, minhashing, and locality-sensitive hashing (LSH) techniques and corresponding algorithms. 

  - Shingling class to generate shingles of length in a given document
  - CompareSets class which computes the Jaccard similarity in two given docs, where each doc contains the hashed shingles.
  - CompareSignature class which computes how similar two vectors are in terms of their vectors components. This is in theory supposed to be close to the Jaccard similarity.
  - main2 module which runs the overall algorithm in only three short documents.
  - main module which runs the overall algorithm over a large corpus of texts

### Installation

Just run the main class and make sure you are inside the lab1 folder. Some packages can be installed
```sh
$ cd lab1
$ python3 -m pip install numpy
$ python3 -m pip install mmh3
$ python3 -m pip install pyspark
```