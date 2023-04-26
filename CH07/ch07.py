# %%

from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
from statsmodels.stats.diagnostic import acorr_ljungbox 
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import numpy as np 
from statsmodels.tsa.stattools import adfuller 
from itertools import product 
from typing import Union 

# %% 
df = pd.read_csv('../data/jj.csv')
df.head()
# %%

fig, ax = plt.subplots() 
ax.plot(df['data'])
ax.set_xlabel('date')
ax.set_ylabel('earning per share')
plt.xticks(np.arange(0, 81, 8), [1960, 1962, 1964, 1966, 1968, 1970, 1972, 1974, 1976, 1978, 1980])

fig.autofmt_xdate()
plt.tight_layout()

plt.savefig('figures/share.png', dpi=300)


# %% adfuller test for stationary series 

adf_fuller_result = adfuller(df['data'])
print(f"ADF statistics: {adf_fuller_result[0]}")
print(f"p-value: {adf_fuller_result[1]}")

# p-value is not < 0.05, = 1.0, series need to be differenced
# %%


eps_diff = np.diff(df['data'], n=1)
ad_fuller_result = adfuller(eps_diff) 
print(f"ADF Statistics: {ad_fuller_result[0]}")
print(f"p-value: {ad_fuller_result[1]}")

# p-value of 0.908 is much higher thatn the confidence level of 5%

# %%

eps_diff2 = np.diff(eps_diff, n=1)
ad_fuller_result = adfuller(eps_diff2) 

print(f"ADF Statistics: {ad_fuller_result[0]}")
print(f"p-value: {ad_fuller_result[1]}")


# finally 0.006 is less than p-value, and we have a large ADF statistics
# the integration order i = 2, since we differemced two times.
# %%

fig, ax = plt.subplots() 
ax.plot(df['date'][2:], eps_diff2)
ax.set_xlabel('date')
ax.set_ylabel('earnings per share - diff (USD)')

#plt.xticks()
# plot this because we dont want every time to be shown
plt.xticks(np.arange(0, 81, 8), [1960, 1962, 1964, 1966, 1968, 1970, 1972, 1974, 1976, 1978, 1980])


fig.autofmt_xdate()
plt.tight_layout()

# thoughts: series that have autocorrelation, violates iid assumption in linear regression. 
# which meas that breush-pagan test is not a good choice to test for homoscedasticity 
# which we uses stationary, agumented dicker fuller test for it. 


# %%
def optimize_ARIMA(endog: Union[pd.Series, list], order_list:list, d:int) -> pd.DataFrame:
    results = [] 

    for order in tqdm(order_list):
        try: 
            model = SARIMAX(endog, order=(order[0], d, order[1]),
                            simple_differencing=False).fit(disp=False)
        except:
            continue 

        aic = model.aic 
        results.append([order, aic])

    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q)', 'AIC']

    # soft in ascending order, lower AIC is better 
    result_df = result_df.sort_values(by="AIC", ascending=True).reset_index(drop=True) 

    return result_df 
# %%
ps = range(0, 4, 1)
qs = range(0, 4, 1)
d = 2

order_list = list(product(ps, qs))
# %%

train =  df['data'][:-4]

result_df = optimize_ARIMA(train, order_list, d) 
print(result_df) 
# %%
model = SARIMAX(train, order=(3,2,3), simple_differencing=False)
model_fit = model.fit(disp=False)

print(model_fit.summary())
# %%
model_fit.plot_diagnostics(figsize=(10,8));

plt.savefig('figures/CH07_F07_peixeiro.png', dpi=300)
# %%
residuals = model_fit.resid

lbvalue, pvalue = acorr_ljungbox(residuals, np.arange(1, 11, 1))

print(pvalue)
# %%
test = df.iloc[-4:]

test['naive_seasonal'] = df['data'].iloc[76:80].values
test

# %%
ARIMA_pred = model_fit.get_prediction(80, 83).predicted_mean

test['ARIMA_pred'] = ARIMA_pred
test
# %%
fig, ax = plt.subplots()

ax.plot(df['date'], df['data'])
ax.plot(test['data'], 'b-', label='actual')
ax.plot(test['naive_seasonal'], 'r:', label='naive seasonal')
ax.plot(test['ARIMA_pred'], 'k--', label='ARIMA(3,2,3)')

ax.set_xlabel('Date')
ax.set_ylabel('Earnings per share (USD)')
ax.axvspan(80, 83, color='#808080', alpha=0.2)

ax.legend(loc=2)

plt.xticks(np.arange(0, 81, 8), [1960, 1962, 1964, 1966, 1968, 1970, 1972, 1974, 1976, 1978, 1980])
ax.set_xlim(60, 83)

fig.autofmt_xdate()
plt.tight_layout()

plt.savefig('figures/CH07_F08_peixeiro.png', dpi=300)

# %%

def mape(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100 

# %% 
mape_naive_seasonal = mape(test['data'], test['naive_seasonal'])
mape_ARIMA = mape(test['data'], test['ARIMA_pred'])
# %%
fig, ax = plt.subplots() 

x = ['naive seasonal', 'ARIMA(3,2,3)']
y = [mape_naive_seasonal, mape_ARIMA]


ax.bar(x, y, width=0.4) 
ax.set_xlabel('models')
ax.set_ylabel('MAPE')
ax.set_ylim(0, 15) 

for index, value in enumerate(y):
    plt.text(x=index, y=value, s=str(round(value,2)), ha='center')

plt.savefig('figures/mape_score_models.png')