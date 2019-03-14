import xgboost as xgb
from xgboost.sklearn import  XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import f1_score
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
def classification(clf):
    acc = 0
    acc_f1 = 0
    for iter in range(1):
        clf.fit(X_train_robust_norm , y_train)
        y_predict = clf.predict(X_test_robust_norm)
        acc += accuracy(y_predict, y_test)
        acc_f1 += f1_score(y_test, y_predict, average='micro')
    print('With augm',acc, acc_f1)
    return [acc, acc_f1]
