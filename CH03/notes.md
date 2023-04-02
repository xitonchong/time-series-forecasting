# the random walk process
the random walk is a process in which there is an equal chance of going up or down by a random number. this is usually observed in financial data. 

Present value y_t is a function of the value at previous timestemp y_{t-1}, a constant C, and a random number \theta_t, also termed white noise.   
\theta_t is the realization of standard normal distribution, which has variance of 1 and mean of 0. 

mathematically, 
y_t = C + y_{t-1} + \theta_t
Note that if the constant C is nonzero, we designate this process as a random walk with drift. 


## 3.2 Identifying a random walk 
A `random walk` is defined as a series whose first difference is stationary and uncorrelatd. 

Stationary - time series whose statistical properties do not change over time. In other words, it has a constant mean, variance, and autocorrelation, and these properties are independent of time. 

Many forecasting models assume stationary. the moving average model, autoregressive model, and autoregressive moving average model all assume stationary.   if data is non-stationary, its properties are going to change over time, which would mean that our model oarameters must also change through time. This means that we cannot possibly derive a function of future values as a function of past values, since the coefficients chnage at each point in time, making forecast unreliable. 

Transformation - mathematical manupulation of data that stabilizes its mean and variance, thus making it stationary.   
 - Differencing: differencing involves calculating the series of changes from one timestep to another. substract the value of previous timestep y_{t-1} from the value in the present y_t to obtain the differenced value. 
    y^{'}_t = y_t - y_{t-1} 
 Applying a log function to the series can stabilize its variance. 

 taking the difference once is applying a `first-order` differencing, second time would be a `second-order` differencing. 

 keep in mind that when we model a time series that has been transformed, we must apply
 `inverse transform` to return the result of the model in original units of measurement. 

## 3.2.2 Testing for Stationary

Augmented Dicker-Fuller (ADF) test 
    helps us determine if a time series is stationary by testing the present of a unit root. if a unit root is present the time series is not stationary. 

    the null hypothesis states that a unit root is present, meaning that our time series is not stationary. 

## 3.2.3 the autocorrelation fuction 
    Once a process is stationary, plotting the ACF is agreat way to understnad whay type of process you are analyzing.  In this case, we will use it to determine if we are studying a random walk a not. 

    - correlations: measures the extend of a linear relationship between two variables. 

    - Autocorrelation: measures the linear relationship between lagged values of a time series.  
        ACF revelas how the correlation between any two values changes as the lag increases.  Here, the lag is simply the number of timesteps separating two values. 
        In other words, it measure the correlation of the time series with itself. 
        When we plot the ACF function, the coefficient is the dependent variable, while the lag is the indepedent variable.  Note that autocorrelation coefficient at lag 0 will always be 1. this make sense intuitively, because the linear relationship between a variable and itself at the same time should be perfect, and therefore equal to 1. 
        In the presence of a trend, a plot of the ACF will show the coefficients are high for short lags, and they will decrease linearly as the lag increases. If the data is seasonal, the ACF plot will also display cyclical patterns.  Therefore, plotting the ACF function of a non-stationary process will not give us more information thatn is available by looking at the evolution of our process through time.  However, plotting the ACF for a stationary process can help us identify the presence of a random walk.



