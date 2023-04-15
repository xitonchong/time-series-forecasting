## Autoregressive process
is a regression of a varialbe against itself.  In a time series, this means that the present value is linearly dependent on its past values.  

The autoregressive process is denoted as AR(p), where p is the order. the general expression of an AR(p) model is
y_t = C + \theta_1 * y_t-1 * \theta_2 * y_t-2 + ...  + white_noise

## 5.3.1. the partial autocorrelation PACF
in formal terms, the partial autocorrelation measure the correlation between lagged alues in a time series when we remove the influence of correlated lagged values in between.  those are known as `confouding variables`.  the partial autocorrelation function will reveal how the partial autocorrelation varies when the lags increases. 

we can plot the partial autocorrelation function to determine the order of a stationary series AR(p) process. the coefficients wll be non-significats after lag p.


### Summary 
 - an autoreggressive process states that the present value is linearly depedent on its past value and an error term. 
 - if the ACF plot of a stationary process shows a slow decay, then you likely have an autoregressive prces.s 
 0 the partial autoregressve measures the correlation between two lagged values of a times series when you remove the ffect of the other autocorrelated lagged values. 
 - plotting the PACF of a stationary autoregressive process will show the order o of the process. the coefficients will be significant up until lag p only. 
 