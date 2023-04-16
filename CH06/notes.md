

the `autoregressive moving average` process is the combination of autoregressive process and the moving average process.  it states that the present value is linearly dependent on its own previous value and a constant, just like in an autoregessive process, as well as on the mean of the series, the current error term, like the moving average process. 

```latex
yt = C + φ1yt–1 + φ2yt–2 +⋅⋅⋅+ φpyt–p + µ + ϵt
 + θ1ϵt–1 + θ2ϵt–2 +⋅⋅⋅+ θqϵt–q
```

An ARMA(0,q) process is equivalent to an MA(q) process, since the order p = 0 cancels the AR(p) portion. An ARMA(p,0) process is equivalent to an AR(p) process, since
the order q = 0 cancels the MA(q) portion.

![ACF plot cannot infer ARMA process](figures/arma_1_1_process.png)
Notice the sinusoidal pattern on the plot, meaning that an AR(p) process is in play. 
also, the alst significant coefficietn is at lag 2, which suggests that q=2. 
however, we know that we simulated an ARMA(1,1) process, so q mus tbe equal to 1!
therefore, the ACF plot cannot be used to infer the order of q of an ARMA(p,q) process. 



![PACF on ARMA process](figures/pacf_on_arma1-1.png)
again we have a sinusoidal pattern with no clear cutoff between significant and non-significant coefficients. from this plot, we canot infer that p=1 in our simulated ARMA(1,1) process, meaning that we canot determine the order o of an ARMA(p,q) process using a PACF plot. 

 
> if your process is stationary and both the ACF and PACF plots show a decaying or sinusoidal pattern, then it is a stionary ARMA(p,q) process. 


## 6.4 Devising a general modeling procedure 


> # Akaike Information Criterion (AIC) 
> the AIC is a measure of the quality of a model in relation to other models. it is used for model selection.
> the AIC is a funciton of the number of parameters k in a model and the maximum value of the likelihood function L: 
>>  AIC = 2k - 2 ln(L)
> the lower value of AIC, the better the model. Selecting the order oto the AIC allow us to keep a balance between the complexity of a model and its goodness of fit to the data. 

SARIMAX(simple_differencing=?)
If simple_differencing = True is used, then the endog and exog data are differenced prior to putting the model in state-space form. This has the same effect as if the user differenced the data prior to constructing the model, which has implications for using the results:

Forecasts and predictions will be about the differenced data, not about the original data. (while if simple_differencing = False is used, then forecasts and predictions will be about the original data).

If the original data has an Int64Index, a new RangeIndex will be created for the differenced data that starts from one, and forecasts and predictions will use this new index.

> Autocorrelation: correlation between the elements of a series and others from the same series separated from them by a given interval.
Autocorrelation, sometimes known as serial correlation in the discrete time case, is the correlation of a signal with a delayed copy of itself as a function of delay.