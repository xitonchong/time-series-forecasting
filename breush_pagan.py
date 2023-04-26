# %% 

import numpy as np 
import pandas as pd 
import statsmodels.formula.api as smf 

#create dataset
df = pd.DataFrame({'rating': [90, 85, 82, 88, 94, 90, 76, 75, 87, 86],
                   'points': [25, 20, 14, 16, 27, 20, 12, 15, 14, 19],
                   'assists': [5, 7, 7, 8, 5, 7, 6, 9, 9, 5],
                   'rebounds': [11, 8, 10, 6, 6, 9, 6, 10, 10, 7]})


# fit regression model 
fit = smf.ols('rating ~ points + assists + rebounds', data=df).fit() 

# view model summary 
print(fit.summary())
# %% Step 2: peform a Breush-Pagan test 

from statsmodels.compat import lzip 
import statsmodels.stats.api as sms 


# perform breush pagan test 
names = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(fit.resid, fit.model.exog)
# %%
print(fit.model.exog) 
print(df.head())
# %%
lzip(names, test) 

# %%
