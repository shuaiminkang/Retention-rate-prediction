


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.feature_selection import SelectKBest

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.neural_network import MLPClassifier
from mlxtend.classifier import StackingCVClassifier

from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV   


#read the train and test dataset
path="/Users/skang/Downloads/2017umassuconn/"

train_cm=pd.read_csv(path+'train_complete.csv')

train_cm_x=train_cm[train_cm.columns[train_cm.columns!='cancel']]

train_cm_y=train_cm['cancel']

test_cm=pd.read_csv(path+'test_complete.csv')


# Stacking models
## 8 level-1 classifiers: KNN, SVC, Logistic, RF, ExtraTree, GBM, XGBoost, MLP
## level-2: Logistic

RANDOM_SEED = 42

k=730
clf1 = KNeighborsClassifier(n_neighbors=k)

clf2 =  XGBClassifier( learning_rate =0.083, n_estimators=158,max_depth=2,
 min_child_weight=4, gamma=0,subsample=0.9, colsample_bytree=0.8,reg_alpha=0.1,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=RANDOM_SEED)


clf_baglogis=BaggingClassifier(LogisticRegression(penalty='l1',C=0.05,random_state=RANDOM_SEED),
                               max_samples=0.6,max_features=0.9,random_state=RANDOM_SEED)  
clf_poly2baglogis=make_pipeline(PolynomialFeatures(degree=2),clf_baglogis)
clf3 = clf_poly2baglogis


gb_params = {
         'n_estimators': 158,
          'max_features': 0.2,
         'max_depth': 2,
         'min_samples_leaf': 2,
         'learning_rate': 0.1,
         'verbose': 0,
         'random_state':RANDOM_SEED
           }
clf4=GradientBoostingClassifier(**gb_params)

svc_params = {
       'kernel' : 'linear',
       'C' : 1.6,
       'random_state':RANDOM_SEED,
      'probability':True
        }
clf5 = SVC(**svc_params)


rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
    'max_features': 6,
    'max_depth': 10,
    'min_samples_leaf': 17,
    'max_features' : 'sqrt',
    'verbose': 0
}
clf6=RandomForestClassifier(**rf_params)

pipeline = Pipeline([('select', SelectKBest(k=13)), ('clf', ExtraTreesClassifier(n_estimators =500,max_features=3,min_samples_leaf=50))])

et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 25,
    'max_features':8,
    'verbose': 0
}
clf7=ExtraTreesClassifier(**et_params)


clf8=MLPClassifier(activation='relu', alpha=0.1, batch_size=350, beta_1=0.9,
       beta_2=0.999, early_stopping=True, epsilon=1e-08,
       hidden_layer_sizes=(50, 50), learning_rate='constant',
       learning_rate_init=0.01, max_iter=500, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=100,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)


lr = BaggingClassifier(LogisticRegression(random_state=RANDOM_SEED,penalty='l1',C=0.1),max_samples=0.8,max_features=0.8)

sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3,clf5,clf7,clf8], use_probas=True,
                            meta_classifier=lr)

sclf.fit(train_cm_x.values, train_cm_y.values)
predict_y=sclf.predict_proba(test_cm.values)[:-1]

df = pd.DataFrame(predict_y)
df.to_csv("predicted_y.csv")
print('5-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3,clf5,clf7,clf8,sclf], 
                      ['KNN', 
                       'Extreme gradient boosting', 
                       'bagging Logistic Regression',
                       'Linear SVC',
                       'Extra Tree',
                       'Neural Network',
                       'StackingClassifier']):

    scores =cross_val_score(clf, train_cm_x.values, train_cm_y.values,cv=5, scoring='roc_auc')
    print("AUC: %0.7f (+/- %0.7f),%0.7f, [%s]" 
          % (scores.mean(), scores.std(), scores.min(),label))
