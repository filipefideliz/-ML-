# %%
import pandas as pd

df = pd.read_excel('../dados/dados_cerveja_nota.xlsx')
df

 # %%
df['Aprovado'] = df['nota'] >=5
df
# %%
from sklearn import linear_model
reg = linear_model.LogisticRegression(penalty=None,
                                      fit_intercept=True)

features  = ['cerveja']
target = 'Aprovado'
## modelo aprende
reg.fit(df[features],df[target])
## modelo preve
reg_predict = reg.predict(df[features])
reg_predict

# %% metric para ver o acerto do modelo
from sklearn import metrics

reg_acc = metrics.accuracy_score(df[target],reg_predict)
reg_acc
# %%
reg_conf = metrics.confusion_matrix(df[target], reg_predict)
reg_conf = pd.DataFrame(reg_conf,
                        index=['False', 'True'],
                        columns=['False', 'True'],) 
reg_conf

# %%
from sklearn import tree

arvore = tree.DecisionTreeClassifier(max_depth=2)

# %%
arvore.fit(df[features], df[target])
arvore_predict = arvore.predict(df[features])
arvore_predict

# %% metric para ver o acerto do modelo
from sklearn import metrics

arvore_acc = metrics.accuracy_score(df[target],arvore_predict)
arvore_acc
# %%
arvore_conf = metrics.confusion_matrix(df[target], arvore_predict)
arvore_conf
# %%
# %%
from sklearn import naive_bayes

nb = naive_bayes.GaussianNB()

# %%
nb.fit(df[features], df[target])
nb_predict = nb.predict(df[features])
nb_predict

# %% metric para ver o acerto do modelo
from sklearn import metrics

nb_acc = metrics.accuracy_score(df[target],nb_predict)
nb_acc
# %%
nb_conf = metrics.confusion_matrix(df[target], nb_predict)
nb_conf
# %%