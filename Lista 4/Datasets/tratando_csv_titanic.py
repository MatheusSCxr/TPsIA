import pandas as pd
import numpy as np


#estou usando o conjunto de treino tanto para gerar, quanto para testar a árvore.
base = pd.read_csv('./train.csv', sep =',', usecols=['Survived', 'Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])


from sklearn.preprocessing import LabelEncoder
#para codificar todos os atributos para laberEncoder de uma única vez
#base_encoded = base.apply(LabelEncoder().fit_transform)
cols_label_encode = ['Embarked'] #único não numérico
base[cols_label_encode] = base[cols_label_encode].apply(LabelEncoder().fit_transform)

## PARA O ID3 - discretizar Age e Fare

# #age em faixas fixas, sem labels
# base['Age'] = pd.cut(base['Age'], bins=[0, 12, 18, 35, 60, 100], labels=False)

# #fare em quartis
# base['Fare'] = pd.qcut(base['Fare'], q=4, labels=False)

from sklearn.preprocessing import OneHotEncoder
cols_onehot_encode = ['Sex']
#oneHotEncoder
onehot = OneHotEncoder(sparse_output=False) #retorna um array denso

#oneHotEncoder apenas nas colunas categóricas
df_onehot = onehot.fit_transform(base[cols_onehot_encode])

#salvar novos nomes das colunas
nomes_das_colunas = onehot.get_feature_names_out(cols_onehot_encode)

#criar dataFrame com os dados codificados e as novas colunas
df_onehot = pd.DataFrame(df_onehot, columns=nomes_das_colunas)

#combinar as colunas codificadas com as colunas que não foram transformadas
base_encoded= pd.concat([df_onehot, base.drop(columns=cols_onehot_encode)], axis=1)




#selecionar a coluna survived
X_prev = base_encoded.drop(columns=['Survived'])
y_classe = base_encoded['Survived']

from sklearn.model_selection import train_test_split

#usando 20% da base com o stratify
X_treino, X_teste, y_treino, y_teste = train_test_split(X_prev, y_classe, test_size = 0.20, random_state = 42, stratify=y_classe)

#criar pickle
import pickle

with open('./titanic_Discret.pkl', mode = 'wb') as f:
  pickle.dump([X_treino, X_teste, y_treino, y_teste], f)