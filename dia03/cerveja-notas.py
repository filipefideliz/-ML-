# %%
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('../dados/dados_cerveja_nota.xlsx')
df
# %%
plt.plot(df['cerveja'], df['nota'], 'o')
plt.grid(True)
plt.title('Relaçao Nota x Cerveja')
plt.ylim(0,11)
plt.xlim(0,11)
plt.xlabel('Cerveja')
plt.ylabel('Nota')
plt.show()

# %% 
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(df[['cerveja']], df['nota'])
# %%
