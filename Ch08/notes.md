

## Seasonal autoregressive integrated moving average (SARIMA) model 
the seasonal autoregresive integrated moving average (SARIMA) model add seasonal parameters to the ARIMA(p,d,q) model. 
It is denoted as SARIMA(p,d,q)(P,D,Q)m where P is the order of the seasonal process, D is the seasonal integration of integration, Q is the order of the seasonal MA(Q) process, and m is the frequency  or the number of observations per seasonal cycle.
Note that a SARIMA(p,d,q)(0,0,0)m model is the equivalent to an ARIMA(p,d,q) model. 

## Time series decomposition 
is a statistical task that separates the ties ereis into its 3 main componens: a trend component, a seasonal compoennt, then the residuals
the trend component represetnt the long-tern change in the time series. this compoentn is responsible for time series that increase or decrease over time. the seasonal component is the periodic pattern in the time series. it represents repeated fluctuation that occurs over a fixed period of time.  finally, the residuals, or the noise express any irregularity that canno tbe explained by the trend or seasonal component. 
