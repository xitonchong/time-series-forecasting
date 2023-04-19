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
import matplotlib.pyplot as plt 

fig, ax = plt.subplots() 

ax.plot(df['hourly_bandwidth'])
ax.set_xlabel('Time')
ax.set_ylabel('Hourly bandwidth usage')


plt.xticks(
    np.arange(0, 10000, 730), 
    ['Jan 2019', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan 2020', 'Feb'])

fig.autofmt_xdate()
plt.tight_layout()

plt.savefig('figures/CH06_F02_peixeiro.png', dpi=300)

# %%
ADF_result = adfuller(df['hourly_bandwidth'])

print(f"ADF Statistics: {ADF_result[0]}")
print(f"p-value: {ADF_result[1]}")

# p is 0.87 >> higher than 0.05 ; need to diff it to make it stationary.
# %%

bandwidth_diff = np.diff(df.hourly_bandwidth, n=1) 

# %%
import matplotlib.pyplot as plt 
# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.plot(bandwidth_diff)
ax.set_xlabel('Time')
ax.set_ylabel('Hourly bandwith usage - diff (MBps)')

plt.xticks(
    np.arange(0, 10000, 730), 
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', '2020', 'Feb'])

fig.autofmt_xdate()
plt.tight_layout()

# plt.savefig('figures/CH06_F_peixeiro.png', dpi=300)
# %%
ADF_result = adfuller(bandwidth_diff)

print(f'ADF Statistic: {ADF_result[0]}')
print(f'p-value: {ADF_result[1]}')
# %%
plot_acf(bandwidth_diff, lags=20);

plt.tight_layout()
# %%
plot_pacf(bandwidth_diff, lags=20);

plt.tight_layout()
# %%
df_diff = pd.DataFrame({'bandwidth_diff': bandwidth_diff})

train = df_diff[:-168]
test = df_diff[-168:]

print(len(train))
print(len(test))
# %%
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 8))

ax1.plot(df['hourly_bandwidth'])
ax1.set_xlabel('Time')
ax1.set_ylabel('Hourly bandwidth usage (MBps)')
ax1.axvspan(9831, 10000, color='#808080', alpha=0.2)

ax2.plot(df_diff['bandwidth_diff'])
ax2.set_xlabel('Time')
ax2.set_ylabel('Hourly bandwidth - diff (MBps)')
ax2.axvspan(9830, 9999, color='#808080', alpha=0.2)

plt.xticks(
    np.arange(0, 10000, 730), 
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', '2020', 'Feb'])

fig.autofmt_xdate()
plt.tight_layout()

plt.savefig('figures/CH06_F17_peixeiro.png', dpi=300)
# %%
from typing import Union

def optimize_ARMA(endog: Union[pd.Series, list], order_list: list) -> pd.DataFrame:
    
    results = []
    
    for order in tqdm_notebook(order_list):
        try: 
            model = SARIMAX(endog, order=(order[0], 0, order[1]), simple_differencing=False).fit(disp=False)
        except:
            continue
            
        aic = model.aic
        results.append([order, aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q)', 'AIC']
    
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df
# %%
ps = range(0, 4, 1)
qs = range(0, 4, 1)

order_list = list(product(ps, qs))
# %%
ps = range(0, 4, 1)
qs = range(0, 4, 1)

order_list = list(product(ps, qs))
# %%
model = SARIMAX(train['bandwidth_diff'], order=(2,0,2), simple_differencing=False)
model_fit = model.fit(disp=False)
print(model_fit.summary())
# %%
model_fit.plot_diagnostics(figsize=(10, 8));

plt.savefig('figures/CH06_F19_peixeiro.png', dpi=300)
# %%
residuals = model_fit.resid

lbvalue, pvalue = acorr_ljungbox(residuals, np.arange(1, 11, 1))

print(pvalue)

# %%  6.6 Forecasting bandwidth usage 

def rolling_forecast(df: pd.DataFrame, train_len: int, horizon: int, window: int, method: str):
    total_len = train_len + horizon 
    end_idx = train_len
    
    if method == 'mean':
        pred_mean = []
        
        for i in range(train_len, total_len, window):
            mean = np.mean(df[:i].values)
            pred_mean.extend(mean for _ in range(window))
            
        return pred_mean

    elif method == 'last':
        pred_last_value = []
        
        for i in range(train_len, total_len, window):
            last_value = df[:i].iloc[-1].values[0]
            pred_last_value.extend(last_value for _ in range(window))
            
        return pred_last_value
    
    elif method == 'ARMA':
        pred_ARMA = []
        
        for i in range(train_len, total_len, window):
            model = SARIMAX(df[:i], order=(2,0,2))
            res = model.fit(disp=False)
            predictions = res.get_prediction(0, i + window - 1)
            oos_pred = predictions.predicted_mean.iloc[-window:]
            pred_ARMA.extend(oos_pred)
            
        return pred_ARMA

# %%
TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 2

pred_mean = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'mean')
pred_last_value = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'last')
pred_ARMA = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'ARMA')

test.loc[:, 'pred_mean'] = pred_mean
test.loc[:, 'pred_last_value'] = pred_last_value
test.loc[:, 'pred_ARMA'] = pred_ARMA

test.head()
fig, ax = plt.subplots()

ax.plot(df_diff['bandwidth_diff'])
ax.plot(test['bandwidth_diff'], 'b-', label='actual')
ax.plot(test['pred_mean'], 'g:', label='mean')
ax.plot(test['pred_last_value'], 'r-.', label='last')
ax.plot(test['pred_ARMA'], 'k--', label='ARMA(2,2)')

ax.legend(loc=2)

ax.set_xlabel('Time')
ax.set_ylabel('Hourly bandwidth - diff (MBps)')

ax.axvspan(9830, 9999, color='#808080', alpha=0.2)

ax.set_xlim(9800, 9999)

plt.xticks(
    [9802, 9850, 9898, 9946, 9994],
    ['2020-02-13', '2020-02-15', '2020-02-17', '2020-02-19', '2020-02-21'])

fig.autofmt_xdate()
plt.tight_layout()

plt.savefig('figures/CH06_F20_peixeiro.png', dpi=300)
# %%
