import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from upc.UnexpectedPatternCoocurrence import UnexpectedPatternCoocurrence

def findAnomalies():
    
    files = []
    docs = []
    pwd = os.path.dirname(__file__)
    for file in os.listdir(pwd):
        if file.endswith(".txt"):
            with open(file, 'r') as f:
                doc=f.read().replace('\n', '')
                files.append(file)
                docs.append(doc)
    
    cv = CountVectorizer(stop_words="english", max_features=1000, analyzer="word", token_pattern="[a-zA-Z]{4,}")
    counts = cv.fit_transform(docs)  
    dArr = counts.data
    dArr[dArr  > 0] = 1
    counts.data = dArr    
        
    idAndDesc = {}    
    for (key,val) in cv.vocabulary_.items():
        idAndDesc[val] = key
    
    upc = UnexpectedPatternCoocurrence(encoded=True)
    upc.fit(counts)
    
    (scores, patterns, maxScore) = upc.computeScores(counts)    
        
    sp = []
    for (i, score) in enumerate(scores):
        pattern = patterns[i]
        sp.append((i, score, pattern)) 
               
    sortedScores = sorted(sp, key=lambda score: score[1], reverse=True)    
    
    for score in sortedScores:
        file = (files[score[0]] + "                    ")[0:20] 
        pattern0 = prettyPrintPattern(score[2], 0, idAndDesc)
        pattern1 = prettyPrintPattern(score[2], 1, idAndDesc)
        print(file, "\t", score[1], "\t", pattern0, "\t", pattern1)

def prettyPrintPattern(rawPatternPair, num, idAndDesc):
    if rawPatternPair is None:
        return "No Anomaly"
    colId = rawPatternPair[num][0]
    prettyDesc = idAndDesc.get(colId, "not-found")
    return ("Id=" + str(colId) + " /desc=" + prettyDesc + "                         ")[0:25]
    
            
if __name__ == '__main__':
    findAnomalies()