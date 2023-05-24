import pandas as pd
import pickle
# Векторизация
from sklearn.feature_extraction.text import TfidfVectorizer


stress = pd.read_csv('stress.csv')
x=stress['clean_text']
y=stress['label']

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(main_data['text'])
with open(f'vectorizer.pickle', 'wb') as handle:
    pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(f'label.pickle', 'wb') as handle:
    pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(f'features.pickle', 'wb') as handle:
    pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)
