pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm

import numpy as np
import pandas as pd

df= pd.read_csv('Stress.csv')

stress=df.copy()

# Регулярные выражения
import re 

# Строка обработки
import string

# NLP инструменты
import spacy

nlp=spacy.load('en_core_web_sm')
from spacy.lang.en.stop_words import STOP_WORDS

# Импорт Natural Language Tool Kit for NLP
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')                                #Multilingual Wordnet Data from OMW
from nltk.stem import WordNetLemmatizer

from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from collections import Counter

# функция предобработки
def preprocess(text,remove_digits=True):
    text = re.sub('\W+',' ', text)                                        # замена символов, не являющихся словами
    text = re.sub('\s+',' ', text)                                        # замена лишних пробелов
    text = re.sub("(?<!\w)\d+", "", text)                                 # удаление всех чисел кроме тех, которые прикреплены к тексту
    text = re.sub("-(?!\w)|(?<!\w)-", "", text)                           # удаление всех дефисов кроме стоящих между двумя словами
    text=text.lower()
    nopunc=[char for char in text if char not in string.punctuation]      # удаление знаков препинания
    nopunc=''.join(nopunc)
    nopunc=' '.join([word for word in nopunc.split() if word.lower() not in stopwords.words('english')])   # удаление стоп-слов
    return nopunc

# функция лемматизации
def lemmatize(words):
   
    words=nlp(words)
    lemmas = []
    for word in words:
        lemmas.append(word.lemma_)
    return lemmas

# конвертация в строку
def listtostring(s):
    str1=' '
    return (str1.join(s))

def clean_text(input):
    word=preprocess(input)
    lemmas=lemmatize(word)
    return listtostring(lemmas)

def split_data(data): # вернет dataframe в котором заданное чилос элементов
  elements = 10
  print("Подготовка данных для тестов")
  print(elements, "данных для тестов!")
  data_testing = data.tail(elements)
  for i in range(data.shape[0]-1, (data.shape[0] - elements), -1):
    data.drop([i], axis = 0, inplace = True)

stress['clean_text']=stress['text'].apply(clean_text)

# Defining target & feature for ML model building
from sklearn.model_selection import train_test_split
x=stress['clean_text']
y=stress['label']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
stress.to_csv('stress.csv', index = False)
x_train.to_csv('x_train.csv')
x_test.to_csv('x_test.csv')
y_train.to_csv('y_train.csv')
y_test.to_csv('y_test.csv')
