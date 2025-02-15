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