import pandas as pd
import numpy as np

#carregar o csv
base = pd.read_csv('./tennis.csv', sep=',')


from sklearn.preprocessing import LabelEncoder

#aplicar labelencoder em todas as colunas categóricas
for col in base.columns:
    base[col] = LabelEncoder().fit_transform(base[col])

#obter todas colunas, menos a classe
cols = base.columns
X_prev = base.drop(columns=['play'])
y_classe = base['play']

from sklearn.model_selection import train_test_split

#usando 20% da base com o stratify
X_treino, X_teste, y_treino, y_teste = train_test_split(X_prev, y_classe, test_size=0.20, random_state=42, stratify=y_classe)

#criar pickle
import pickle
with open('./tennis.pkl', mode='wb') as f:
    pickle.dump([X_treino, X_teste, y_treino, y_teste], f)