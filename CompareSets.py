import pdb
class CompareSets():

    def jaccard_sim(self, doc1, doc2):
        #pdb.set_trace()
        # To compute similarity we first need the union and intersection  
        #pdb.set_trace()
        #doc1 = set(doc1)
        #doc2 = set(doc2)
        #union = doc1 + doc2
        union = doc1.union(doc2)
        intersection = doc1.intersection(doc2)
        #ersection = list(set(doc1) & set(doc2))
        print("Similarity between the 2 sets is " + str(round(len(intersection)/len(union), 2)))
        #print("Similarity between the 2 sets is " + str(round(len(intersection)/(len(union)-len(intersection)), 2)))
