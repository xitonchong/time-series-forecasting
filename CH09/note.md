

# adding external variales to our model 

## SARIMAX model 
the sarimax model simply adds a linear combination of exogeneous varaibles to the SARIMA model.  this allows us to model the impact of external variables onthe future value of a time series. 

the sarimax model is the most general model for forecasting time series.  you can see that if you have no seasonal patterns, it becomes an ARIMAX model.  


## caveat for using SARIMAX

what if you need to predict 2 timesteps into the futures? while this is possible with a SARIMA model, the SARIMAX model requires us to forecast the exogeneous variables too. 

