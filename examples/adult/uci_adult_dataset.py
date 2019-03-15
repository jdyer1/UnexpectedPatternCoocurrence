import os
import csv
import pandas as pd
from upc.UnexpectedPatternCoocurrence import UnexpectedPatternCoocurrence

def findAnomalies():
    
    idAndDesc = loadDictionary()
    filePath = os.path.join(os.path.dirname(__file__), "adult.csv")
    df = pd.read_csv(filePath, header=None)
    upc = UnexpectedPatternCoocurrence(encoded=False)
    upc.fit(df)
    
    (scores, patterns, maxScore) = upc.computeScores(df)
    
    print("max score ", maxScore)
    print("-----")
    
    sp = []
    for (i, score) in enumerate(scores):
        pattern = patterns[i]
        pattern0 = prettyPrintPattern(pattern, 0, idAndDesc)
        pattern1 = prettyPrintPattern(pattern, 1, idAndDesc)
        sp.append((i, score, pattern0, pattern1, pattern)) 
               
    sortedScores = sorted(sp, key=lambda score: score[1], reverse=True)
    
    
    print("Top 100 scores (most anomalous):")
    for (j, score) in enumerate(sortedScores):
        print(score)
        if j==100:
            break
        
    print("-----")    
    print("Anomalies found with 'predict':")
    is_inlier = upc.predict(df)
    for (i, val) in enumerate(is_inlier):
        if val == -1:
            print("anomaly found ", i)
    

def prettyPrintPattern(rawPatternPair, num, idAndDesc):
    if rawPatternPair is None:
        return "No Anomaly"
    attrId = rawPatternPair[num][0]
    prettyDesc = idAndDesc.get(attrId, "not-found")
    return "attrId=" + str(attrId) + "/desc=" + prettyDesc

def loadDictionary():
    with open("dictionary.csv", mode="r") as infile:
        reader = csv.reader(infile)        
        _dict = {int(rows[0]):rows[1] for rows in reader}
    return _dict
    

if __name__ == '__main__':
    findAnomalies()
    

