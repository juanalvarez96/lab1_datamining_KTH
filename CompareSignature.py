import pdb
class CompareSignature():

    def comparator(self, col1, col2):
        intersect = 0
        for i in range (0, len(col1)):
            if col1[i]==col2[i]: 
                intersect = intersect + 1
        union = len(col1) 
        return intersect/union
