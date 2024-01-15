import numpy as np

class VectorSimilarity():
    '''
    Class for vector similarity search.
    
    - x, y: equal length numpy arrays
    '''
    
    def __init__(self, x: np.array, y: np.array) -> None:

        self.x = x
        self.y = y
        self._validate_params()
        
    def _validate_params(self):
        if isinstance(self.x, np.ndarray) and isinstance(self.y, np.ndarray) and len(self.x) == len(self.y):
            pass
        else:
            raise TypeError('Input arrays must be equal length ndarrays!')
            
    def euclidean(self):
        '''Returns the euclidean distance between the two vectors'''
        return np.sqrt(np.sum(np.square((self.x - self.y))))
    
    def manhattan(self):
        '''Returns the manhattan distance between the two vectors'''
        return np.sum(np.absolute(self.x - self.y))
    
    def cosine(self):
        '''Returns the cosine similarity between the two vectors'''

        num = np.sum(self.x*self.y)
        denom = np.sqrt(np.sum(np.square(self.x)))*np.sqrt(np.sum(np.square(self.y)))
        cos_theta = num/denom

        return cos_theta