# %%
import pandas as pd

df = pd.read_excel("../dados/dados_frutas.xlsx")
df
# %% Como descobrir as caracteristicas desta planilha

filtro_arredondada= df["Arredondada"] == 1
filtro_suculenta= df["Suculenta"] == 1
filtro_vermelha= df["Vermelha"] == 1
filtro_doce= df["Doce"] == 1

df[filtro_arredondada & filtro_doce & filtro_suculenta & filtro_vermelha ]

# %% Como podemos fazer a maquina aprender

from sklearn import tree

features = ['Arredondada','Suculenta','Vermelha','Doce']
target = 'Fruta'

X= df[features]
y= df[target]
# %%

arvore = tree.DecisionTreeClassifier()
arvore.fit(X,y)

# %%
import matplotlib.pyplot as plt

plt.figure(dpi=700)

tree.plot_tree( arvore,
                class_names=arvore.classes_,
                feature_names=features,
                filled=True
                )