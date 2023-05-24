import pandas as pd


stress = pd.read_csv('stress.csv')
x=stress['clean_text']
y=stress['label']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

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


# Перекрестная проверка, чтобы избежать переобучения
import statistics as st
vector = TfidfVectorizer()

x_train_v = vector.fit_transform(x_train)
x_test_v  = vector.transform(x_test)

# Построение модели
lr =LogisticRegression()
mnb=MultinomialNB()
dct=DecisionTreeClassifier(random_state=1)
knn=KNeighborsClassifier()
rf=RandomForestClassifier(random_state=1)
ab=AdaBoostClassifier(random_state=1)
m  =[lr,mnb,dct,knn,rf,ab]
model_name=['Logistic R','MultiNB','DecTRee','KNN','R forest','Ada Boost']

results, mean_results, p, f1_test=list(),list(),list(),list()

# Подгонка модели, перекрестная проверка и оценка производительности
def algor(model):
    print('\n',i)
    pipe=Pipeline([('model',model)])
    pipe.fit(x_train_v,y_train)
    cv=StratifiedKFold(n_splits=5)
    n_scores=cross_val_score(pipe,x_train_v,y_train,scoring='f1_weighted',cv=cv,n_jobs=-1,error_score='raise') # As it is an Imbalance data so f1 score will give a better accuracy than normal accuracy.
    results.append(n_scores)
    mean_results.append(st.mean(n_scores))
    print('f1-Score(train): mean= (%.3f), min=(%.3f)) ,max= (%.3f), stdev= (%.3f)'%(st.mean(n_scores), min(n_scores), max(n_scores),np.std(n_scores)))
    y_pred=cross_val_predict(model,x_train_v,y_train,cv=cv)
    p.append(y_pred)
    f1=f1_score(y_train,y_pred, average = 'weighted')
    f1_test.append(f1)
    print('f1-Score(test): %.4f'%(f1))

for i in m:
    algor(i)


x=stress['clean_text']
y=stress['label']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

vector = TfidfVectorizer()
x_train = vector.fit_transform(x_train)
x_test = vector.transform(x_test)
model_lr_tf=LogisticRegression()

model_lr_tf.fit(x_train,y_train)
y_pred=model_lr_tf.predict(x_test)
# Model Evaluation
    
conf=confusion_matrix(y_test,y_pred)
acc_lr=accuracy_score(y_test,y_pred)
f1_lr=f1_score(y_test,y_pred,average='weighted')

print('Accuracy: ',acc_lr)
print('F1 Score: ',f1_lr)
print(10*'===========')
print('Confusion Matrix: \n',conf)
print(10*'===========')
print('Classification Report: \n',classification_report(y_test,y_pred))