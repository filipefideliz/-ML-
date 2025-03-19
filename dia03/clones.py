# %%
import pandas as pd

df = pd.read_parquet('../dados/dados_clones.parquet')
df

# %%
df.groupby(["Status "])[['Estatura(cm)', 'Massa(em kilos)']].mean()
# %%
df['bool'] = df['Status '] == 'Apto'
df

# %%

features = [
    "Estatura(cm)",
    "Massa(em kilos)",
    "Distância Ombro a ombro",
    "Tamanho do crânio",
    "Tamanho dos pés",
]

cat_features = ["Distância Ombro a ombro",
                "Tamanho do crânio",
                "Tamanho dos pés"]

X = df[features]

# %%
# Transformação de categorias para Numérico
from feature_engine import encoding
onehot = encoding.OneHotEncoder(variables=cat_features)
onehot.fit(X)
X = onehot.transform(X)
X

# %%

from sklearn import tree
arvore = tree.DecisionTreeClassifier(max_depth=3)
arvore.fit(X, df["Status "])

# %%

import matplotlib.pyplot as plt
plt.figure(dpi=600)
tree.plot_tree(arvore,
               class_names=arvore.classes_,
               feature_names=X.columns,
               filled=True,
               )
# %%
