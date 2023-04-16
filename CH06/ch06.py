## Modeling complex time series 


# %% 
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_process import ArmaProcess 
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm 
from itertools import product 


import matplotlib.pyplot as plt  
import numpy as np 
import pandas as pd 

import warnings 
warnings.filterwarnings('ignore')

# %% Identifying a stationary ARMA process 


np.random.seed(1) 

ar1 = np.array([1, -0.33])
ma1 = np.array([1, 0.9])

ARMA_1_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=1000)

# %%
ADF_result = adfuller(ARMA_1_1)
print(f"ADF statistics: {ADF_result[0]}")
print(f"p-value: {ADF_result[1]}")

# %%
plot_acf(ARMA_1_1, lags=20)
plt.tight_layout() 
plt.savefig('figures/arma_1_1_process.png')

# notice that there
# %%
plot_pacf(ARMA_1_1, lags=20)
plt.tight_layout() 
plt.savefig('figures/pacf_on_arma1-1.png')
# %% 6.4.1 Selecting the best model 

ps = range(0, 4, 1) 
qs = range(0, 4, 1) 

order_list = list(product(ps, qs)) 
print(order_list) 

# %%
from typing import Union 

def optimize_ARMA(endog: Union[pd.Series, list], order_list: list) -> pd.DataFrame: 
    results = [] 

    for order in tqdm(order_list): 
        try: 
            model = SARIMAX(endog, order=(order[0], 0, order[1]),
                            simple_differencing=False).fit(disp=False)
        except: 
            continue 
        aic = model.aic 
        results.append([order, aic])

    result_df = pd.DataFrame(results) 
    result_df.columns = ['(p,q)', 'AIC']

    # soft in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)

    return result_df 

# %%
result_df = optimize_ARMA(ARMA_1_1, order_list) 

print(result_df)
# %% 6.4.3 performing residual analysis 

# Qualitative aalysis: studyin gthe Q-Q plot 
from statsmodels.graphics.gofplots import qqplot 
gamma = np.random.default_rng().standard_gamma(shape=2, size=1000)

qqplot(gamma, line='45')
plt.savefig('figures/qqplot.png')

# %%

normal = np.random.normal(size=1000)
qqplot(normal, line='45')
plt.savefig('figures/theoretical_quantiles.png')

# %% 6.4.4 Performing residual analysis 

model = SARIMAX(ARMA_1_1, order=(1,0,1), simple_differencing=False)
model_fit = model.fit(disp=False) 
residuals = model_fit.resid
# %%

from statsmodels.graphics.gofplots import qqplot

qqplot(residuals, line='45')
plt.savefig('figures/qqplot_on_arma11.png')
# %%

model_fit.plot_diagnostics(figsize=(10, 8))
plt.savefig('figures/residual_plot_diag.png')
# %%
from statsmodels.stats.diagnostic import acorr_ljungbox

df = acorr_ljungbox(residuals, np.arange(1, 11, 1)) 

acorr_test = (df.lb_pvalue.values>=0.05).all()
print(f"autocorrelation test passed?: {acorr_test}")
# %%  6.5 Applying the general mdeling procedure 
import pandas as pd 


df = pd.read_csv('../data/bandwidth.csv')
df.head()
# %%
