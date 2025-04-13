# %% 
import pandas as pd
from sklearn import model_selection

df = pd.read_csv('../dados/dados_pontos.csv',sep=';')
df

# %%

features = df.columns[3:-1]
target = 'flActive'

X_train,X_test,y_train,y_test = model_selection.train_test_split(df[features],
                                                                df[target], 
                                                                test_size=0.2,
                                                                random_state=42,
                                                                stratify=df[target])


print('Tx treino reposta: ',y_train.mean())
print('Tx Reposta teste: ', y_test.mean())
# %% verificando os valores faltantes
X_test.isna().sum().T
# %% prenchendo od dados faltantes com o valor maximo
input_avgRecorrencia = X_train['avgRecorrencia'].max()

X_train['avgRecorrencia'] = X_train['avgRecorrencia'].fillna(input_avgRecorrencia)

X_test['avgRecorrencia'] = X_test['avgRecorrencia'].fillna(input_avgRecorrencia)

# %% aqui a gente treina 
from sklearn import metrics
from sklearn  import tree

arvore = tree.DecisionTreeClassifier(max_depth=5,
                                     min_samples_leaf=50,
                                     random_state=42)
arvore.fit(X_train, y_train)

# %% preve na base na propria base
tree_pred_train = arvore.predict(X_train)
tree_acc_train = metrics.accuracy_score(y_train, tree_pred_train)
print('Arvore Train Acc: ',tree_acc_train)

tree_pred_test = arvore.predict(X_test)
tree_acc_test = metrics.accuracy_score(y_test, tree_pred_test)
print('Arvore Test Acc: ',tree_acc_test)

tree_proba_train = arvore.predict_proba(X_train)[:,1]
tree_acc_train = metrics.roc_auc_score(y_train, tree_proba_train)
print("Árvore Train AUC:", tree_acc_train)

# Aqui a gente prevê na base de teste
tree_proba_test = arvore.predict_proba(X_test)[:,1]
tree_acc_test = metrics.roc_auc_score(y_test, tree_proba_test)
print("Árvore Test AUC:", tree_acc_test)
