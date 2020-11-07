
import binascii  # Needed for the hash
import pdb  # Debugging

'''
Initialize class Shingling with k = 5, since we have short texts.
'''
class Shingling():
    def __init__(self, k):
        self.k = k
        self.shinglings = list()
        self.shinglings_hash = list()
    
    def clean(self):
        self.shinglings = list()
        self.shinglings_hash = list()

    def hash_func (self, shingle):
        a=0
        mod = 2**32+1
        for word in shingle:
            a=a+hash(word)%mod
        return a

    
    def just_charachters(self, doc):
        filter_chars = [' ', '.', '-', '´', ',', '(', ')', '‘', '’']
        text = doc.content
        filtered_text = ''.join((filter(lambda x: x not in filter_chars, text)))
        '''
        for i in range(len(filtered_text)-self.k+1):
            pdb.set_trace()
            shingle = words[i:i+self.k]

        '''

    def make_shingles(self, doc):
        # We receive a row type from main class
       
        words = doc.split(' ')
        for i in range(len(words)-self.k+1):
            shingle = words[i:i+self.k]
            
            #pdb.set_trace()
            # Compute hash for that shingle
            hash_val = self.hash_func(shingle)

            # Add hash to list
            if ((hash_val not in self.shinglings_hash) and (shingle not in self.shinglings)):
                self.shinglings_hash.append(hash_val)
                self.shinglings.append(shingle)


