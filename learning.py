import pandas as pd
import numpy as np
import joblib


# Построение модели
from sklearn.model_selection import GridSearchCV,StratifiedKFold,KFold,train_test_split,cross_val_score,cross_val_predict
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier,RandomForestClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

# Оценка модели
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,f1_score,precision_score
from sklearn.pipeline import Pipeline


with open('vectorizer_old.pickle', 'rb') as handle:
    vectorizer = pickle.load(handle)
with open('features_old.pickle', 'rb') as handle:
    X = pickle.load(handle)
with open('vectorizer_old.pickle', 'rb') as handle:
    y = pickle.load(handle)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Самоопределяющаяся функция для преобразования данных в векторную форму с помощью векторизатора tf idf, а также для классификации и создания модели с помощью логистической регрессии.
def model_lr_tf(x_train, x_test, y_train, y_test):

    ovr = LogisticRegression()
    
    ovr.fit(x_train, y_train)
    with open(f'model_lr_tf.pickle', 'wb') as file:
        pickle.dump(ovr, file)

# Самоопределяющаяся функция для преобразования данных в векторную форму с помощью векторизатора tf idf, а также для классификации и создания модели с помощью MultinomialNB.
def model_nb_tf(x_train, x_test, y_train, y_test):
    
    ovr = MultinomialNB()

    ovr.fit(x_train, y_train)
    with open(f'model_nb_tf.pickle', 'wb') as file:
        pickle.dump(ovr, file)


# Самоопределяющаяся функция для преобразования данных в векторную форму с помощью векторизатора tf idf, а также для классификации и создания модели с помощью Decision Tree
def model_dt_tf(x_train, x_test, y_train, y_test):
    
    ovr = DecisionTreeClassifier(random_state=1)
    
    ovr.fit(x_train, y_train)
    with open(f'model_dt_tf.pickle', 'wb') as file:
        pickle.dump(ovr, file)
    

# Самоопределяющаяся функция для преобразования данных в векторную форму с помощью векторизатора tf idf, а также для классификации и создания модели с помощью KNN.
def model_knn_tf(x_train, x_test, y_train, y_test):
    
    ovr = KNeighborsClassifier()
    
    ovr.fit(x_train, y_train)
    with open(f'model_knn_tf.pickle', 'wb') as file:
        pickle.dump(ovr, file)

# Самоопределяющаяся функция для преобразования данных в векторную форму с помощью векторизатора tf idf, а также для классификации и создания модели с помощью Random Forest.
def model_rf_tf(x_train, x_test, y_train, y_test):

    ovr = RandomForestClassifier(random_state=1)

    ovr.fit(x_train, y_train)
    with open(f'model_rf_tf.pickle', 'wb') as file:
        pickle.dump(ovr, file)

# Самоопределяющаяся функция для преобразования данных в векторную форму с помощью векторизатора tf idf, а также для классификации и создания модели с помощью Adaptive Boosting.
def model_ab_tf(x_train, x_test, y_train, y_test):

    ovr = AdaBoostClassifier(random_state=1)

    ovr.fit(x_train, y_train)
    with open(f'model_ab_tf.pickle', 'wb') as file:
        pickle.dump(ovr, file)