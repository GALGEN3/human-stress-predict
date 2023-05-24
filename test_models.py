import pandas as pd
import pickle

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

# Время
from time import time

with open('vectorizer.pickle', 'rb') as handle:
vectorizer = pickle.load(handle)
with open('features.pickle', 'rb') as handle:
X = pickle.load(handle)
with open('label.pickle', 'rb') as handle:
y = pickle.load(handle)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Самоопределяющаяся функция для преобразования данных в векторную форму с помощью векторизатора tf idf, а также для классификации и создания модели с помощью логистической регрессии.
def model_lr_tf(x_train, x_test, y_train, y_test):
    global acc_lr_tf,f1_lr_tf,time_lr_tf
    with open('model_lr_tf.pickle', 'rb') as handle:
        ovr = pickle.load(handle)
    # встраивание данных обучения в модель и прогнозирование
    t0 = time()

    y_pred = ovr.predict(x_test)
    
    # Оценка модели
    conf=confusion_matrix(y_test,y_pred)
    acc_lr_tf=accuracy_score(y_test,y_pred)
    f1_lr_tf=f1_score(y_test,y_pred,average='weighted')
    time_lr_tf=time()-t0
    print('Time :',time_lr_tf)
    print('Accuracy: ',acc_lr_tf)
    print(10*'===========')
    print('Confusion Matrix: \n',conf)
    print(10*'===========')
    print('Classification Report: \n',classification_report(y_test,y_pred))
    
    return y_test,y_pred,acc_lr_tf

# Самоопределяющаяся функция для преобразования данных в векторную форму с помощью векторизатора tf idf, а также для классификации и создания модели с помощью MultinomialNB.
def model_nb_tf(x_train, x_test, y_train, y_test):
    global acc_nb_tf,f1_nb_tf,time_nb_tf
    with open('model_nb_tf.pickle', 'rb') as handle:
        ovr = pickle.load(handle)
    # встраивание данных обучения в модель и прогнозирование
    t0 = time()

    y_pred = ovr.predict(x_test)
    
    # Оценка модели
    conf=confusion_matrix(y_test,y_pred)
    acc_nb_tf=accuracy_score(y_test,y_pred)
    f1_nb_tf=f1_score(y_test,y_pred,average='weighted')
    time_nb_tf=time()-t0
    print('Time : ',time_nb_tf)
    print('Accuracy: ',acc_nb_tf)
    print(10*'===========')
    print('Confusion Matrix: \n',conf)
    print(10*'===========')
    print('Classification Report: \n',classification_report(y_test,y_pred))
    
    return y_test,y_pred,acc_nb_tf

# Самоопределяющаяся функция для преобразования данных в векторную форму с помощью векторизатора tf idf, а также для классификации и создания модели с помощью Decision Tree
def model_dt_tf(x_train, x_test, y_train, y_test):
    global acc_dt_tf,f1_dt_tf,time_dt_tf
    with open('model_dt_tf.pickle', 'rb') as handle:
        ovr = pickle.load(handle)
    
    # встраивание данных обучения в модель и прогнозирование
    t0 = time()

    y_pred = ovr.predict(x_test)
    
    # Оценка модели    
    conf=confusion_matrix(y_test,y_pred)
    acc_dt_tf=accuracy_score(y_test,y_pred)
    f1_dt_tf=f1_score(y_test,y_pred,average='weighted')
    time_dt_tf=time()-t0
    print('Time : ',time_dt_tf)
    print('Accuracy: ',acc_dt_tf)
    print(10*'===========')
    print('Confusion Matrix: \n',conf)
    print(10*'===========')
    print('Classification Report: \n',classification_report(y_test,y_pred))
    
    return y_test,y_pred,acc_dt_tf

# Самоопределяющаяся функция для преобразования данных в векторную форму с помощью векторизатора tf idf, а также для классификации и создания модели с помощью KNN.
def model_knn_tf(x_train, x_test, y_train, y_test):
    global acc_knn_tf,f1_knn_tf,time_knn_tf
    with open('model_knn_tf.pickle', 'rb') as handle:
        ovr = pickle.load(handle)

    # встраивание данных обучения в модель и прогнозирование
    t0 = time()

    y_pred = ovr.predict(x_test)
    
    # Оценка модели    
    conf=confusion_matrix(y_test,y_pred)
    acc_knn_tf=accuracy_score(y_test,y_pred)
    f1_knn_tf=f1_score(y_test,y_pred,average='weighted')
    time_knn_tf=time()-t0
    print('Time : ',time_knn_tf)
    print('Accuracy: ',acc_knn_tf)
    print(10*'===========')
    print('Confusion Matrix: \n',conf)
    print(10*'===========')
    print('Classification Report: \n',classification_report(y_test,y_pred))

# Самоопределяющаяся функция для преобразования данных в векторную форму с помощью векторизатора tf idf, а также для классификации и создания модели с помощью Random Forest.
def model_rf_tf(x_train, x_test, y_train, y_test):
    global acc_rf_tf,f1_rf_tf,time_rf_tf
    with open('model_rf_tf.pickle', 'rb') as handle:
        ovr = pickle.load(handle)

    # встраивание данных обучения в модель и прогнозирование
    t0 = time()

    y_pred = ovr.predict(x_test)
    
    # Оценка модели    
    conf=confusion_matrix(y_test,y_pred)
    acc_rf_tf=accuracy_score(y_test,y_pred)
    f1_rf_tf=f1_score(y_test,y_pred,average='weighted')
    time_rf_tf=time()-t0
    print('Time : ',time_rf_tf)
    print('Accuracy: ',acc_rf_tf)
    print(10*'===========')
    print('Confusion Matrix: \n',conf)
    print(10*'===========')
    print('Classification Report: \n',classification_report(y_test,y_pred))

# Самоопределяющаяся функция для преобразования данных в векторную форму с помощью векторизатора tf idf, а также для классификации и создания модели с помощью Adaptive Boosting.
def model_ab_tf(x_train, x_test, y_train, y_test):
    global acc_ab_tf,f1_ab_tf,time_ab_tf
    with open('model_ab_tf.pickle', 'rb') as handle:
        ovr = pickle.load(handle)

    # встраивание данных обучения в модель и прогнозирование
    t0 = time()

    y_pred = ovr.predict(x_test)
    
    # Оценка модели    
    conf=confusion_matrix(y_test,y_pred)
    acc_ab_tf=accuracy_score(y_test,y_pred)
    f1_ab_tf=f1_score(y_test,y_pred,average='weighted')
    time_ab_tf=time()-t0
    print('Time : ',time_ab_tf)
    print('Accuracy: ',acc_ab_tf)
    print(10*'===========')
    print('Confusion Matrix: \n',conf)
    print(10*'===========')
    print('Classification Report: \n',classification_report(y_test,y_pred))

# Оценка моделей
print('********************Logistic Regression*********************')
print('\n')
model_lr_tf(x_train, x_test, y_train, y_test)
print('\n')
print(30*'==========')
print('\n')
print('********************Multinomial NB*********************')
print('\n')
model_nb_tf(x_train, x_test, y_train, y_test)
print('\n')
print(30*'==========')
print('\n')
print('********************Decision Tree*********************')
print('\n')
model_dt_tf(x_train, x_test, y_train, y_test)
print('\n')
print(30*'==========')
print('\n')
print('********************KNN*********************')
print('\n')
model_knn_tf(x_train, x_test, y_train, y_test)
print('\n')
print(30*'==========')
print('\n')
print('********************Random Forest Bagging*********************')
print('\n')
model_rf_tf(x_train, x_test, y_train, y_test)
print('\n')
print(30*'==========')
print('\n')
print('********************Adaptive Boosting*********************')
print('\n')
model_ab_tf(x_train, x_test, y_train, y_test)
print('\n')
print(30*'==========')
print('\n')

# Создание табулярного формата для наглядного сравнения
tbl=pd.DataFrame()
tbl['Model']=pd.Series(['Logistic Regreesion','Multinomial NB','Decision Tree','KNN','Random Forest','Adaptive Boosting'])
tbl['Accuracy']=pd.Series([acc_lr_tf,acc_nb_tf,acc_dt_tf,acc_knn_tf,acc_rf_tf,acc_ab_tf])
tbl['F1_Score']=pd.Series([f1_lr_tf,f1_nb_tf,f1_dt_tf,f1_knn_tf,f1_rf_tf,f1_ab_tf])
tbl['Time']=pd.Series([time_lr_tf,time_nb_tf,time_dt_tf,time_knn_tf,time_rf_tf,time_ab_tf])
tbl.set_index('Model')

# Лучшая модель по версии F1 Score
tbl.sort_values('F1_Score',ascending=False)

tbl.to_csv('results.csv', index = False)
