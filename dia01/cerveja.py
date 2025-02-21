# %%  
import pandas as pd

df = pd.read_excel('../dados/dados_cerveja.xlsx')
df

# %%

df_R =df.replace({'mud': 0, 'pint':1,
                 'não':0,'sim':1 ,
                 'escura':1, 'clara':0})
df_R

# %%
from sklearn import tree

features = ['temperatura', 'copo', 'espuma', 'cor']
target = 'classe'

X = df_R[features]
y = df_R[target]

# %%

arvore = tree.DecisionTreeClassifier()
arvore.fit(X,y)
# %%
import matplotlib.pyplot as plt

plt.figure(dpi=600)
tree.plot_tree( arvore,
                class_names=arvore.classes_,
                feature_names=features,
                filled=True
                )

# %%
probas = arvore.predict_proba([[0,1,1,1]])[0]
pd.Series(probas, index=arvore.classes_)