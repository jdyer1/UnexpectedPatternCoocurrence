import numbers
import warnings

import scipy.sparse
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_array

INTEGER_TYPES = (numbers.Integral, np.integer)

class UnexpectedPatternCoocurrence(BaseEstimator):
    '''
    Unsupervised Outlier Detection using Unexpected Pattern Co-Occurences (UPC).
    
    The anomaly score of each sample is called the Unexpected Pattern Co-Occurence 
    Score, alternatively known as the Beauty and Brains (BnB) Score.  Anomalous 
    samples are determined by finding unexpected feature combinations.  This 
    estimator can deal directly with categorical data without the need to
    perform one-hot encoding. 
    
    This implementation examines feature pairs as its item-set and scores each
    example based on the description in section 3.3 of the paper.
    
    This implementation does NOT look at all feature combinations nor does it select 
    the most interesting candidates using an algorithm such as SLIM, as done in the 
    reference implementation.  See here:  http://eda.mmci.uni-saarland.de/prj/upc/
    
    Parameters
    ----------
    encoded : boolean, optional (default=True)
        Is the data one-hot encoded?
            - if True, the combination of column position plus value represents 
            a unique entity.
            - if False, the value alone represents a unique entity
            - In both cases, zeros are ignored
    
    threshold : int or float, optional (default=.9)
        The minimum score needed to determine an outlyer.
            - If int, then score >= threshold determines outlier status.
            - If float, then score >= threshold * maxScore determines outlier status.

    References
    ----------
    .. [1] Bertens, Roel, Vreeken, Jiles and Siebes, Arno (2017)
            Efficently Discovering Unexpected Pattern Co-Occurences. SDM17.
            
    .. [2] Bertens, Roel, Vreeken, Jiles and Siebes, Arno (2016, Feb)
           Beauty and Brains: Detecting Anomalous Pattern Co-Ocurrences.
           Technical Report 1512.07048v2.
    '''

    def __init__(self, threshold=.9, encoded=True):
        self._patternCounts = { }
        self._numExamples = 0
        
        if not isinstance(threshold, INTEGER_TYPES) and not (0. < threshold <= 1.):
            raise ValueError("threshold must be an integer or a float in (0, 1], got %r" % threshold)
        
        self.threshold = threshold    
        self.encoded = encoded   
            
    def fit(self, X, y=None):
        """Fit estimator.  Calling this multiple times adds additional 
        training examples to those already seen.  This extracts the patterns
        to consider and counts the occurrences of each.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input samples. Accepts categorical attributes, for which
            use ``int`` to represent.

        Returns
        -------
        self : object
            Returns self.
        """
        if y is not None:
            y = check_array(y, accept_sparse=True, ensure_2d=False, allow_nd=True, ensure_min_samples=0, ensure_min_features=0)
            if len(y)>0:
                warnings.warn("This estimator works on unlabelled data; Y is ignored", RuntimeWarning)
        
        X = check_array(X, accept_sparse=['csr'])
        self._numExamples = self.countExamples(X)
        self.computePatternCounts(X) 
        return self   
    
    def predict(self, X):
        """Predict if a particular sample is an outlier or not.  This computes
        a UPC/BnB score of each example and populates the "scores" and
        "maxScore" attributes.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input samples. 

        Returns
        -------
        is_inlier : array, shape (n_samples,)
            For each observations, tells whether or not (+1 or -1) it should
            be considered as an inlier.
        """        
        (scores, patterns, maxScore) = self.computeScores(X)      
        
        if isinstance(self.threshold, INTEGER_TYPES):
            minScore = self.threshold
        else:
            minScore = self.threshold * maxScore
            
        is_inlier = np.ones(X.shape[0], dtype=int)
        for (index, score) in enumerate(scores):
            is_inlier[index] = -1 if score >= minScore else 1 
                       
        return is_inlier

    
    def decision_function(self, X):
        """maximum anomaly score of each pattern.

        The anomaly score of an input sample is computed as
        the maximum of each pattern's UPC (or BnB) score.


        Parameters
        ----------
        X : The training input samples. 

        Returns
        -------
        scores : array containing the anomaly score of the 
            input samples. The higher, the more abnormal.

        """              
        (scores, patterns, maxScore) = self.computeScores(X)
        return scores
         
    def computeScores(self, X): 
        """Same as decision_function, but also returns the most 
        anomalous pattern for each example.
        
        Parameters
        ----------
        X : The training input samples. 

        Returns
        -------
        (scores, patterns, maxScore) : The elements contain:
            - scores: an array with anomaly scores
            - patterns: an array of 2-element tuples with the 
                most anomalous pattern combination
            - maxScore: the maximum score among the examples        
        """
        if self._numExamples < 1:
            raise ValueError("Must call 'fit' first.")
        
        X = check_array(X, accept_sparse=['csr'])
              
        if(scipy.sparse.issparse(X)):
            (scores, patterns, maxScore) = self.computeScoresSparse(X)
        else:
            (scores, patterns, maxScore) = self.computeScoresDense(X)                
        return (scores, patterns, maxScore)
                
    def computeScoresSparse(self, csrMatrix):
        scores = []
        patterns = []
        maxScore = 0
        for row in csrMatrix:
            rowMaxScore = 0
            rowMaxPattern = None
            endIndex = len(row.indices) - 1
            for (i, idx) in enumerate(row.indices):
                if i == endIndex:
                    break
                dataX = row.data[i]
                X = (idx, dataX) if self.encoded else (dataX,)
                for i1 in range((i+1), (endIndex+1)):
                    dataY = row.data[i1]
                    idxY = row.indices[i1]
                    Y = (idxY, dataY) if self.encoded else (dataY,)
                    (score, XY) = self.bnbScore(X, Y)
                    if score > rowMaxScore:
                        rowMaxScore = score
                        rowMaxPattern = XY  
                                          
            if rowMaxScore > maxScore:
                maxScore = rowMaxScore   
                  
            scores.append(rowMaxScore)
            patterns.append(rowMaxPattern)
            
        return (scores, patterns, maxScore)
    
    def computeScoresDense(self, denseArr):
        scores = []
        patterns = []
        maxScore = 0
        for row in denseArr:
            rowMaxScore = 0
            rowMaxPattern = None
            endIndex = row.shape[0] 
            for idx in range(0, endIndex):                
                dataX = row[idx]
                if dataX is not None and dataX != 0:
                    X = (idx, dataX) if self.encoded else (dataX,)
                    for idxY in range((idx+1), endIndex):
                        dataY = row[idxY]
                        if dataY is not None and dataY != 0:
                            Y = (idxY, dataY) if self.encoded else (dataY,)
                            (score, XY) = self.bnbScore(X, Y)
                            if score > rowMaxScore:
                                rowMaxScore = score
                                rowMaxPattern = XY   
                                                 
            if rowMaxScore > maxScore:
                maxScore = rowMaxScore 
                    
            scores.append(rowMaxScore)
            patterns.append(rowMaxPattern)      
            
        return (scores, patterns, maxScore)
    
    def computePatternCounts(self, X):
        if(scipy.sparse.issparse(X)):
            self.computePatternCountsSparse(X)
        else:
            self.computePatternCountsDense(X)
                
    def computePatternCountsSparse(self, X):
        for row in X:
            endIndex = len(row.indices) - 1
            for (i, idx) in enumerate(row.indices):
                data = row.data[i]
                key = (idx, data) if self.encoded else (data,)
                self.addPattern(key)
                if i < endIndex:
                    for i1 in range((i+1), (endIndex+1)):
                        data1 = row.data[i1]
                        idx1 = row.indices[i1]
                        key1 = (key, (idx1, data1)) if self.encoded else (key, (data1,))
                        self.addPattern(key1)
    
    def computePatternCountsDense(self, X):
        for row in X:
            endIndex = row.shape[0] - 1
            for idx in range(0, (endIndex + 1)):
                val = row[idx]
                if val != None and val != 0:
                    key = (idx, val) if self.encoded else (val,)
                    self.addPattern(key)
                    if idx < endIndex:
                        for idx1 in range((idx+1), (endIndex + 1)):
                            val1 = row[idx1]
                            if val1 != None and val1 != 0:
                                key1 = (key, (idx1, val1)) if self.encoded else (key, (val1,))
                                self.addPattern(key1)

    def addPattern(self, key):                   
        val = self._patternCounts.get(key)
        if val is None:
            val = 0
        val += 1;
        self._patternCounts[key] = val   
    def countExamples(self, X):        
        if(scipy.sparse.issparse(X)):
            return len(X.indptr) - 1
        else:
            return X.shape[0]    

    def bnbScore(self, X, Y):
        XY = (X, Y)    
            
        pXY = self._patternCounts.get(XY, 0) / self._numExamples
        pX = self._patternCounts.get(X, 0) / self._numExamples
        pY = self._patternCounts.get(Y, 0) / self._numExamples
        
        log2pxy = 0 if pXY == 0 else np.log2(pXY)
        log2px = 0 if pX == 0 else np.log2(pX)
        log2py = 0 if pY == 0 else np.log2(pY)
            
        # sec 3.3:  score = max of -log2(P(XY)) + log2(P(X) * P(Y))
        score = -1 * (log2pxy - log2px - log2py)        
                        
        return (score, XY)
    
        