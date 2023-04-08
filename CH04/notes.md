
# Defining a moving average process
A `moving average` process, states that the current value is linearly dependent on the current and past error terms.  

The errors terms are assumed to be mutually indepedent and normally distributed. 

general expression of a MA(q) model, where q is the order: 
|
    y_t = mean + error_t + theta * error_t-1 + ... + theta_q * error_t-q
|

## Forecasting using the MA(q) model 
Wen using an MA(q) model, forecasting beyond q steps into the future will simply return the mean, because there are no error terms to estimate beyond q steps. we can use rolling forecasts to predict up to q steps at a time in order avoid predicting only the mean of the series. 

