# %% 
from statsmodels.tsa.seasonal import seasonal_decompose, STL 
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt  
import numpy as np 
import pandas as pd 

import warnings 
warnings.filterwarnings('ignore')

# %%
df = pd.read_csv('../data/GOOGL.csv')
df.head()
# %%

fig, ax = plt.subplots() 
ax. plot(df['Date'], df['Close'])
ax.set_xlabel('Date')
ax.set_ylabel('Closing price (USD)')

plt.xticks(
    [4, 24, 46, 68, 89, 110, 132, 152, 174, 193, 212, 235], 
    ['May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan 2021', 'Feb', 'Mar', 'April'])

fig.autofmt_xdate()
plt.tight_layout()

plt.savefig('figures/CH03_F01_peixeiro.png', dpi=300)

# %% 3.1 the random walk process 

np.random.seed(42) 
steps = np.random.standard_normal(1000)
steps[0] = 0
random_walk = np.cumsum(steps) # remember that y_t = C + \theta + y_t-1

fig, ax = plt.subplots() 
ax.plot(random_walk) 
ax.set_xlabel('timesteps')
ax.set_ylabel('Value')

plt.tight_layout() 
plt.savefig('figures/random_walk.png', dpi=300)


# %% 3.2 Testing for stationary

def simulate_process(is_stationary: bool) -> np.array: 
    np.random.seed(42) 
    process = np.empty(400) 

    if is_stationary: 
        alpha = 0.5 
        process[0] = 0 
    else:
        alpha = 1
        process[0] = 10 
    
    for i in range(400):
        if i+1 < 400: 
            process[i+1] = alpha*process[i] + \
                  np.random.standard_normal()
        else: 
            break
    return process 

# %%
stationary = simulate_process(True) 
non_stationary = simulate_process(False)
# %%
fig, ax = plt.subplots() 

ax.plot(stationary, linestyle='-', label='stationary')
ax.plot(non_stationary, linestyle="--", label='non-stationary')
ax.set_label("timesteps")
ax.set_ylabel("value")
ax.legend(loc=2) 
plt.tight_layout() 
plt.savefig('figures/stationary_vs_non-stationary_series.png', dpi=300)



# %%
 
def mean_over_time(process: np.array) -> np.array:
    mean_func = [] 

    for i in range(len(process)):
        mean_func.append(np.mean(process[:i]))

    return mean_func

# %%
stationary_mean = mean_over_time(stationary)
non_stationary_mean = mean_over_time(non_stationary) 

# %%
fig, ax = plt.subplots() 

ax.plot(stationary_mean, label='stationary')
ax.plot(non_stationary_mean, label="non-stationary", linestyle="--")
ax.set_xlabel('Timesteps')
ax.set_ylabel('Mean')
ax.legend(loc=1) 

plt.tight_layout() 
plt.savefig('figures/mean_stationary_vs_non_stationary.png', dpi=300)
# %%


def var_over_time(process: np.array) -> np.array: 
    var_func = []

    for i in range(len(process)):
        var_func.append(np.var(process[:i]))

    return var_func 

# %%
stationary_var = var_over_time(stationary)
non_stationary_var = var_over_time(non_stationary)
# %%
fig, ax = plt.subplots()

ax.plot(stationary_var, label='stationary')
ax.plot(non_stationary_var, linestyle='--', label='non-stationary')
ax.set_xlabel('Timesteps')
ax.set_ylabel('Variance')
ax.legend(loc=2)
ax.set_title("var over time: stationary vs non-stationary")

plt.tight_layout()
plt.savefig('figures/var_over_time.png', dpi=300)
# %%  Putting it all together 

adf_result = adfuller(random_walk)
print(f"ADF statistics: {adf_result[0]}")
print(f'p-value: {adf_result[1]}')

''' p-value need to be less than 0.05 to reject null hypothesis that there
is a unit root present in the series'''
# %%
plot_acf(random_walk, lags=20) 

plt.tight_layout() 
plt.savefig('figures/autocorrelation_plot.png', dpi=300)

# %%
diff_random_walk = np.diff(random_walk, n=1) 
plt.plot(diff_random_walk) 

plt.title('Differenced random walk')
plt.xlabel('Timesteps')
plt.ylabel("Value")
plt.tight_layout()

plt.savefig('figures/diff_random_walk.png', dpi=300)
# %%
ADF_result = adfuller(diff_random_walk)

print(f'ADF Statistic: {ADF_result[0]}')
print(f'p-value: {ADF_result[1]}')

'''
ADF Statistic: -31.7893108575606
p-value: 0.0'''
# %%
plot_acf(diff_random_walk, lags=20)

plt.tight_layout()

plt.savefig('figures/CH03_F11_peixeiro.png', dpi=300)
# %% 3.3 Forecasting a random walk 


### 3.3.1 forecasting on a long horizon 
df = pd.DataFrame({'value': random_walk})


train = df[:800]
test = df[800:]

fig, ax = plt.subplots() 
ax.plot(random_walk) 
ax.set_xlabel('Timesteps')
ax.set_ylabel('Value')
ax.axvspan(800, 1000, color="#808080", alpha=0.2)
plt.tight_layout()

# %%


mean = np.mean(train.value)
test.loc[:, 'pred_mean'] = mean 
print(test.head())

last_value = train.iloc[-1].value 
test.loc[:, 'pred_last'] = last_value 
print(test.head()) 

deltaX = 800 -1 
deltaY = last_value -1 
drift = deltaY / deltaX

x_vals = np.arange(801, 1001, 1) 
pred_drift = x_vals * drift 
test.loc[:, 'pred_drift'] = pred_drift 

print(test.head())



# %%
fig, ax = plt.subplots()

ax.plot(train.value, 'b-')
ax.plot(test['value'], 'b-')
ax.plot(test['pred_mean'], 'r-.', label='Mean')
ax.plot(test['pred_last'], 'g--', label='Last value')
ax.plot(test['pred_drift'], 'k:', label='Drift')

ax.axvspan(800, 1000, color='#808080', alpha=0.2)
ax.legend(loc=2)

ax.set_xlabel('Timesteps')
ax.set_ylabel('Value')

plt.tight_layout()
plt.savefig('figures/CH03_F15_peixeiro.png', dpi=300)
# %%
from sklearn.metrics import mean_squared_error

mse_mean = mean_squared_error(test.value, test.pred_mean) 
mse_last = mean_squared_error(test.value, test.pred_last) 
mse_drift = mean_squared_error(test.value, test.pred_drift) 


# %%
fig, ax = plt.subplots() 

x = ['mean', 'last value', 'drift']
y = [mse_mean, mse_last, mse_drift]

ax.bar(x, y, width=0.4) 
ax.set_xlabel('Methods')
ax.set_ylabel('MSE')
ax.set_ylim(0, 500) 

for index, value in enumerate(y):
    plt.text(x=index, y=value+5, s=str(round(value, 2)), ha='center')

plt.tight_layout() 
plt.savefig('figures/mse_baseline.png')

# %%
df_shift = df.shift(periods=1)

df_shift.head()


''' this section is to tell us that using recent value is useful for next prediction, 
else will contain too many noise. '''
fig, ax = plt.subplots()

ax.plot(df, 'b-', label='actual')
ax.plot(df_shift, 'r-.', label='forecast')

ax.legend(loc=2)

ax.set_xlabel('Timesteps')
ax.set_ylabel('Value')

plt.tight_layout()

plt.savefig('figures/CH03_F18_peixeiro.png', dpi=300)
mse_one_step = mean_squared_error(test['value'], df_shift[800:])

print('mse one step ', mse_one_step)

fig, ax = plt.subplots()

ax.plot(df, 'b-', label='actual')
ax.plot(df_shift, 'r-.', label='forecast')

ax.legend(loc=2)

ax.set_xlim(900, 1000)
ax.set_ylim(15, 28)

ax.set_xlabel('Timesteps')
ax.set_ylabel('Value')

plt.tight_layout()

plt.savefig('figures/CH03_F19_peixeiro.png', dpi=300)