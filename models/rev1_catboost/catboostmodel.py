import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import pickle
import os

def modelAccuracy(modelname, predictions, test_gs):
    mcounter = 0
    for i in range(len(predictions)):
        if(predictions[i]!=test_gs[i]):
            mcounter+=1
    print(str(modelname)+" accuracy: "+str((1-(mcounter/len(predictions)))))
    return (1-(mcounter/len(predictions)))

def main():
    data = pd.read_csv("data.csv",header=None)
    data = data.sample(frac=1)
    data = data.reset_index(drop=True)
    labels = data[0]
    data = data.drop(0,axis='columns')

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.50, random_state=42)
    if not os.path.isfile("cbm.pk1"):
        cbm = CatBoostClassifier()
        cbm.fit(X_train,y_train)
        with open("cbm.pk1","wb") as file:
            pickle.dump(cbm,file)
    else:
        with open("cbm.pk1","rb") as file:
            cbm = pickle.load(file)
    pred_test = cbm.predict(X_test)
    modelAccuracy("CatBoost", pred_test, y_test.values.tolist())
    soft_preds = cbm.predict_proba(X_test)[:,1]
    print("CatBoost ROC-AUC: "+ str(roc_auc_score(y_test.values.tolist(),soft_preds)))
    return

if __name__ == '__main__':
    main()
    