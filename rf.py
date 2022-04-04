from typing import AsyncIterable
import numpy as np
from dtree import *
from sklearn.metrics import r2_score

class RandomForest621:
    def __init__(self, n_estimators=10, oob_score=False):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan
        
       

    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indexes of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        
        oob = []
        trkr = []
        size = len(X)
        index = np.arange(size)

        for tree in self.trees:
            train_index = np.unique(np.random.choice(size,size,replace=True))
            test_index = index[np.isin(index,np.unique(train_index)) == False]
            tree.fit(X[train_index],y[train_index])

            if self.oob_score:
                pred = np.zeros(len(X))
                trk = np.zeros(len(X))
                y_pred_oob = [leaf.prediction for leaf in tree.predict(X[test_index])]
                y_leaf_oob = [leaf.n for leaf in tree.predict(X[test_index])]
                for i in range(len(test_index)):
                    pred[test_index[i]] = y_pred_oob[i]
                    trk[test_index[i]] = y_leaf_oob[i]
                oob.append(pred)
                trkr.append(trk)

        if self.oob_score:
            oob = np.array(oob).T
            trkr = np.array(trkr).T
            emp = []
            for i,row in enumerate(trkr):
                if np.unique(row).any() == [0.]:
                    emp.append(i)
            
            oob = np.delete(oob,emp,axis=0)
            trkr = np.delete(trkr,emp,axis=0)

            self.oob_score_ = self.oob(oob,np.delete(y,emp),trkr)



class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)

        self.trees = [RegressionTree621(min_samples_leaf=min_samples_leaf,max_features=max_features) for _ in range(n_estimators)]

    def predict(self, X_test) -> np.ndarray:

        preds = np.zeros(len(X_test))
        weight = np.zeros(len(X_test))

        for tree in self.trees:
            leaves=tree.predict(X_test)

            w_preds = []
            lens = []
            
            for leaf in leaves:
                w_preds.append(leaf.prediction)
                lens.append(leaf.n)

            preds += np.array(w_preds)*np.array(lens)
            weight += np.array(lens)


        return preds/weight

        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of samples in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
    def oob(self,oob,y,trkr):
        
        
        sum = np.sum(oob*trkr,axis=1)
        den = np.sum(trkr,axis=1)
        den[den == 0]=1
        return r2_score(y,sum/den)
        
    def score(self, X_test, y_test) -> float:

        y_pred = self.predict(X_test)

        y_bar = np.mean(y_test)
        sst = np.sum((y_test-y_bar)**2)
        sse = np.sum((y_test-y_pred)**2)
        return 1-(sse/sst)
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        

class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)

        self.trees = [ClassifierTree621(min_samples_leaf=min_samples_leaf,max_features=max_features) for _ in range(n_estimators)]
        
    def predict(self, X_test) -> np.ndarray:
        
        all_pred=[]
        
        for tree in self.trees:
            leaves = tree.predict(X_test)

            pred = []
            for leaf in leaves:
                pred.append(leaf.prediction)
            all_pred.append(pred)
        all_pred = np.array(all_pred)

        reslt = [max(list(ar),key = list(ar).count) for ar in all_pred.T[0]]
        # print(type(all_pred.T[0][7]))
        return reslt
        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        y_pred = self.predict(X_test)


        return (y_pred == y_test).sum()/len(y_test)    

    def oob(self,oob,y,trkr):

        unq = np.unique(y)
        reslt = []
        for i in range(len(oob)):
            predictor = {}
            for val in unq:
                predictor[val] = 0
                ind = np.where(oob[i]==val)
                comb = np.sum(trkr[i][ind])
                predictor[val] = comb
            reslt.append(max(predictor, key=predictor.get))


        return (np.array(reslt)==np.array(y)).sum()/len(y)    