import os
import csv
import numpy as np
import scipy.sparse
import pandas as pd
from sklearn.utils.estimator_checks import check_estimator
from upc.UnexpectedPatternCoocurrence import UnexpectedPatternCoocurrence

fileName = "adult_extracted.csv"

def test_compatibility():
    check_estimator(UnexpectedPatternCoocurrence)


def test_csr_matrix_one_hot_encoded():   
    '''
    Note the data is not one-hot encoded, we still
    get the same results either way because leftmost columns
    resemble ohe data
    '''    
    mat = load_data_matrix(fileName, True, True)
    assert scipy.sparse.issparse(mat), "Expected a sparse matrix"   
    anomaly_scores = find_scores(mat, True)
    do_assert(anomaly_scores)

def test_csr_matrix_not_encoded():       
    mat = load_data_matrix(fileName, True, False)
    assert scipy.sparse.issparse(mat), "Expected a sparse matrix"   
    anomaly_scores = find_scores(mat, False)
    do_assert(anomaly_scores)

def test_dense_matrix_one_hot_encoded():
    mat = load_data_matrix(fileName, False, True)
    assert not scipy.sparse.issparse(mat), "Expected a dense matrix"
    anomaly_scores = find_scores(mat, True)
    do_assert(anomaly_scores)

def test_dense_matrix_not_encoded():
    mat = load_data_matrix(fileName, False, False)
    assert not scipy.sparse.issparse(mat), "Expected a dense matrix"
    anomaly_scores = find_scores(mat, False)
    do_assert(anomaly_scores)

def test_dataframe_one_hot_encoded():
    df = load_data_pandas(fileName, True)
    anomaly_scores = find_scores(df, True)
    do_assert(anomaly_scores)

def test_dataframe_not_encoded():
    df = load_data_pandas(fileName, False)
    anomaly_scores = find_scores(df, False)
    do_assert(anomaly_scores)
   
def do_assert(anomaly_scores):
    assert len(anomaly_scores) == 1, "with " + fileName + " expected 1 anomaly."
    
    index = anomaly_scores[0][0]
    score = anomaly_scores[0][1]
    description = anomaly_scores[0][2]
    
    assert index == 16, "with " + fileName + " expected the 17th example to have the anomaly."
    assert score > 4.19 and score < 4.20, "with " + fileName + " expected the anomaly to score 4.196."
    assert description[0][0]==14 and description[1][0]==15, "Expected the anomaly to be the combination of attr 13=Husband and 14=Female."  
    
def find_scores(X, one_hot_encoded):
    upc = UnexpectedPatternCoocurrence(encoded=one_hot_encoded)
    upc.fit(X)
    results = upc.predict(X)
    (scores, patterns, maxScore) = upc.computeScores(X)
    anomaly_scores = []
    maxScore1 = 0
    for (i, result) in enumerate(results):
        score = scores[i]
        if score > maxScore1:
            maxScore1 = score
        if result==-1:
            anomaly_score = (i, score, patterns[i])     
            anomaly_scores.append(anomaly_score)
    assert np.abs(maxScore - maxScore1) < .01, "Expected to get the same max score."
    return anomaly_scores    
    
def load_data_pandas(fileName, one_hot_encoded):
    df = pd.DataFrame(load_data_matrix(fileName, False, one_hot_encoded))
    return df

def load_data_matrix(fileName, return_sparse, one_hot_encoded):
    indptr = [0]
    indices = []
    data = []
    filePath = os.path.join(os.path.dirname(__file__), fileName)
    
    with open(filePath, mode="r") as f:
        r = csv.reader(f, delimiter=",")
        
        for row in r:
            for (i, feature) in enumerate(row):
                
                index = feature if one_hot_encoded else i
                value = 1 if one_hot_encoded else feature
                
                indices.append(index)
                data.append(value)
                
            indptr.append(len(indices))  
                      
    mat = scipy.sparse.csr_matrix((data, indices, indptr), dtype=np.int8)
    
    if not return_sparse:
        mat = mat.todense()    
    
    return mat

