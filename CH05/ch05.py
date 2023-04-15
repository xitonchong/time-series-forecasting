# %% 

from statsmodels.tsa.statespace.sarimax import SARIMAX 
from statsmodels.tsa.arima_process import ArmaProcess 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

# %%
df = pd.read_csv('../data/foot_traffic.csv')

print(df.head())

# %%
fig, ax = plt.subplots()
ax.plot(df['foot_traffic'])
ax.set_xlabel('Time')
ax.set_ylabel('Average weekly foot traffic')

plt.xticks(np.arange(0, 1000, 104), np.arange(2000,2020,2))
fig.autofmt_xdate()
plt.tight_layout() 

plt.savefig('figures/foot_traffic.png')


# %%
ADF_result = adfuller(df.foot_traffic)
print(f"ADF Statstics: {ADF_result[0]}")
print(f"p-value: {ADF_result[1]}")


## need to differentiate another time, p-value >> 0.05 and low ADF stats score
# %%
foot_traffic_diff = np.diff(df.foot_traffic, n=1)
# %%
fig, ax = plt.subplots()

ax.plot(foot_traffic_diff)
ax.set_xlabel('Time')
ax.set_ylabel('Average weekly foot traffic (differenced)')

plt.xticks(np.arange(0, 1000, 104), np.arange(2000, 2020, 2))

fig.autofmt_xdate()
plt.tight_layout()

plt.savefig('figures/CH05_F05_peixeiro.png', dpi=300)
# %%
ADF_result = adfuller(foot_traffic_diff)
print(f"ADF Statstics: {ADF_result[0]}")
print(f"p-value: {ADF_result[1]}")
# p-value < 0.05 and -5 ADF statistcs, series is stationary 

# %%
plot_acf(foot_traffic_diff, lags=20)
plt.tight_layout() 

plt.savefig('figures/autocorrelation_present.png')
# %%


## Simulation 
from statsmodels.tsa.arima_process import ArmaProcess
import numpy as np 


np.random.seed(42)

ma2 = np.array([1,0,0])
ar2 = np.array([1, -0.33, -0.5])

AR2_process = ArmaProcess(ar2, ma2).generate_sample(nsample=1000)

# %% partial autocorrelation 

from statsmodels.graphics.tsaplots import plot_pacf 

plot_pacf(AR2_process, lags=20)
plt.tight_layout() 
plt.savefig('figures/pacf_on_AR_process.png')
# %%

df_diff = pd.DataFrame({'foot_traffic_diff': foot_traffic_diff})
train = df_diff[:-52]
test = df_diff[-52:]

print(len(train))
print(len(test))


# %%
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10,8))

ax1.plot(df['foot_traffic'])
ax1.set_xlabel('Time')
ax1.set_ylabel('Avg. weekly foot traffic')
ax1.axvspan(948, 1000, color='#808080', alpha=0.2)

ax2.plot(df_diff['foot_traffic_diff'])
ax2.set_xlabel('Time')
ax2.set_ylabel('Diff. avg. weekly foot traffic')
ax2.axvspan(947, 999, color='#808080', alpha=0.2)

plt.xticks(np.arange(0, 1000, 104), np.arange(2000, 2020, 2))

fig.autofmt_xdate()
plt.tight_layout()

plt.savefig('figures/CH05_F09_peixeiro.png', dpi=300)

# %% 


def rolling_forecast(df: pd.DataFrame, train_len: int, 
        horizon: int, window: int,  method: str) -> list:
    
    total_len = train_len + horizon
    end_idx = train_len
    
    if method == 'mean':
        pred_mean = [] 

        for i in range(train_len, total_len, window):
            mean = np.mean(df[:i].values)
            pred_mean.extend(mean for _ in range(window))
        return pred_mean 
    
    elif method=='last':
        pred_last_value = [] 

        for i in range(train_len, total_len, window):
            last_value = df[:i].iloc[-1].values[0]
            pred_last_value.extend(last_value for _ in range(window))

        return pred_last_value
    
    elif method == 'AR':
        pred_AR = [] 

        for i in range(train_len, total_len, window):
            model = SARIMAX(df[:i], order=(3,0,0))
            res = model.fit(disp=False)
            predictions = res.get_prediction(0, i + window  -1) 
            oos_pred = predictions.predicted_mean.iloc[-window:]
            pred_AR.extend(oos_pred)

        return pred_AR


# %%
TRAIN_LEN = len(train) 
HORIZON = len(test) 
WINDOW = 1

pred_mean = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'mean')
pred_last_value = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'last')
pred_AR = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'AR')

test['pred_mean'] = pred_mean 
test['pred_last_value'] = pred_last_value 
test['pred_AR'] = pred_AR 

print(test.head())
# %%


fig, ax = plt.subplots() 
ax.plot(df_diff['foot_traffic_diff'])
ax.plot(test['foot_traffic_diff'], 'b-', label='actual')
ax.plot(test['pred_mean'], 'g:', label='mean')
ax.plot(test['pred_last_value'], 'r-.', label='last')
ax.plot(test['pred_AR'], 'k--', label='AR')


ax.legend(loc=2) 

ax.set_xlabel('time')
ax.set_ylabel('Diff. avg. weekly foot traffic')

ax.set_xlim(920, 999)
ax.axvspan(947, 998, color="808080", alpha=0.2)
ax.xticks([936, 988], [2018, 2019])

fig.autofmt_xdate() 
plt.tight_layout()
plt.savefig('figures/prediction_vs_actual_AR.png')

# %%

from sklearn.metrics import mean_squared_error

mse_mean = mean_squared_error(test['foot_traffic_diff'],  test['pred_mean'])
mse_last = mean_squared_error(test['foot_traffic_diff'], test['pred_last_value'])
mse_AR = mean_squared_error(test['foot_traffic_diff'], test['pred_AR'])


print(mse_mean, mse_last, mse_AR) 

# %%
fig, ax = plt.subplots()

x = ['mean', 'last_value', 'AR(3)']
y = [mse_mean, mse_last, mse_AR]

ax.bar(x, y, width=0.4)
ax.set_xlabel('Methods')
ax.set_ylabel('MSE')
ax.set_ylim(0, 5)

for index, value in enumerate(y):
    plt.text(x=index, y=value+0.25, s=str(round(value, 2)), ha='center')

plt.tight_layout()
# %%
df['pred_foot_traffic'] = pd.Series() 
df['pred_foot_traffic'][948:] = df['foot_traffic'].iloc[948] + test['pred_AR'].cumsum()


# %%
fig, ax = plt.subplots() 
ax.plot(df['foot_traffic'], 'b-', label='actual')
ax.plot(df['pred_foot_traffic'], 'k-',  label='AR(3)')

ax.legend(loc=2) 
ax.set_xlabel('Time')
ax.set_ylabel('Average weekly foot traffic')

ax.axvspan(948, 1000, color="#808080", alpha=0.2)

ax.set_xlim(920, 1000) 
ax.set_ylim(650, 770)

plt.xticks([936, 988], [2018, 2019])
fig.autofmt_xdate() 
plt.tight_layout() 

