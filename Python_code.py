# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 01:23:50 2018

@author: VARUN
"""

########################### Problem set 3 Qns 3 ############################
####################### Varun Varadaraj Kandasamy ##########################
############################ vxk161230 ############################



# Necessary Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import kpss

######## Step 1 #########
# Below is the link for bitcoin price
# https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20140101&end=20180619


######## Step 2 #########
# From Fred Euro, Oil, Gold, S&P500 date are collected and merged using Excel


######## Step 3 #########

# Bitcoin data
data1 = pd.read_csv('bitcoin.csv') # importing the bitcoin data
# Euro, Oil, Gold, S&P500 data
data2 = pd.read_csv('fred.csv') # importing the fred data which has euro oil gold sp500
# Innrer join 
''' Make sure in R we have to order the date orelse the Linear Regression will be wrong'''
bitcoin = pd.merge(data1, data2, how='inner',on='Date')
bitcoin


######## Step 4 #########

# Plot Series in python
bitcoin['bprice'].plot()
bitcoin['sp'].plot()
bitcoin['gold'].plot()
bitcoin['oil'].plot()
bitcoin['euro'].plot()


# This below is the normal way(x,y axis scatter plot) to do it but it takes lot of time
# =============================================================================
# p1 = plt.plot(bitcoin['Date'],bitcoin['bprice'])
# p2 = plt.plot(bitcoin['Date'],bitcoin['sp'])
# p3 = plt.plot(bitcoin['Date'],bitcoin['oil'])
# p4 = plt.plot(bitcoin['Date'],bitcoin['euro'])
# p5 = plt.plot(bitcoin['Date'],bitcoin['gold'])
# =============================================================================


######## Step 5 #########

# here for Euro alone we no need to use log, but even if you used its not a big problem.
model5 = smf.ols('np.log(bprice)~np.log(sp)+np.log(oil)+np.log(euro)+np.log(gold)',data=bitcoin).fit()
model5.summary()
# Apart from gold, all the other variables are significant so its total bullshit relationship


######## Step 6 #########

# Always set the stationary to Null hypothesis 
# H0 : stationary
# H1 : Non stationary

# we have to always check for larger pvalue (0.1), 
# so that we can prove its insignificant and not reject null hypothesis

# bprice
kpss(bitcoin['bprice'], regression='c', lags=None) # checking for level stationary
kpss(bitcoin['bprice'], regression='ct', lags=None) # checking for trend stationary
# Taking first differencing
kpss(bitcoin['bprice'].diff().dropna(), regression='c', lags=None) # p-value 0.1
kpss(bitcoin['bprice'].diff().dropna(), regression='ct', lags=None) # p-value 0.1
# level and Trend stationary proved with first differencing.

# gold
kpss(bitcoin['gold'], regression='c', lags=None) # checking for level stationary
kpss(bitcoin['gold'], regression='ct', lags=None) # checking for trend stationary
# Taking first differencing
kpss(bitcoin['gold'].diff().dropna(), regression='c', lags=None) # p-value 0.1
kpss(bitcoin['gold'].diff().dropna(), regression='ct', lags=None) # p-value 0.1
# level and Trend stationary proved with first differencing.

# sp
kpss(bitcoin['sp'], regression='c', lags=None) # checking for level stationary
kpss(bitcoin['sp'], regression='ct', lags=None) # checking for trend stationary
# Taking first differencing
kpss(bitcoin['sp'].diff().dropna(), regression='c', lags=None) # p-value 0.01
kpss(bitcoin['sp'].diff().dropna(), regression='ct', lags=None) # p-value 0.1
# Trend stationary proved with first differencing, but its not level stationary

# oil
kpss(bitcoin['oil'], regression='c', lags=None) # checking for level stationary
kpss(bitcoin['oil'], regression='ct', lags=None) # checking for trend stationary
# Taking first differencing
kpss(bitcoin['oil'].diff().dropna(), regression='c', lags=None) # p-value 0.1
kpss(bitcoin['oil'].diff().dropna(), regression='ct', lags=None) # p-value 0.1
# level and Trend stationary proved with first differencing.

# euro
kpss(bitcoin['euro'], regression='c', lags=None) # checking for level stationary
kpss(bitcoin['euro'], regression='ct', lags=None) # checking for trend stationary
# Taking first differencing
kpss(bitcoin['euro'].diff().dropna(), regression='c', lags=None) # p-value 0.1
kpss(bitcoin['euro'].diff().dropna(), regression='ct', lags=None) # p-value 0.1
# level and Trend stationary proved with first differencing.


######## Step 7 #########

# bprice, gold, sp, oil, euro (1st diffrencing)
model7 = smf.ols('np.log(bprice).diff()~np.log(sp).diff()+np.log(oil).diff()+np.log(euro).diff()+np.log(gold).diff()',data=bitcoin).fit()
model7.summary()
bitcoin.head()

bitcoin['dbprice']=np.log(bitcoin['bprice']).diff().dropna()
bitcoin['dgold']=np.log(bitcoin['gold']).diff().dropna()
bitcoin['dsp']=np.log(bitcoin['sp']).diff().dropna()
bitcoin['doil']=np.log(bitcoin['oil']).diff().dropna()
bitcoin['deuro']=np.log(bitcoin['euro']).diff().dropna()

# Another method for doing it (Professor's Version)
# =============================================================================
# xdata = pd.concat((bitcoin['dgold'],bitcoin['dsp'],bitcoin['doil'],bitcoin['deuro']),1)
# y = bitcoin['dbprice']
# xdata = sm.add_constant(xdata)
# y = y.values[1:]
# xdata = xdata.values[1:,]
# np.linalg.solve(xdata.T.dot(xdata),xdata.T.dot(y))
# 
# sm.OLS(y.values[1:],xdata.values[1:,]).fit().summary()
# xdata.h
# model7_1 = smf.ols('np.log(dbprice)~np.log(dsp)+np.log(doil)+np.log(deuro)+np.log(dgold)',data=bitcoin).fit()
# model7_1.summary()
# =============================================================================


######## Step 8 #########

# The new bitcoin dataset has been formed. From now on we will be having only this dataset
bitcoin = bitcoin[bitcoin['Date'] >="2017-01-01"]
bitcoin['bprice'].plot()
# plot8 = plt.plot(bitcoin['Date'],bitcoin['bprice'])


######## Step 9 #########

# acf and pacf gives the relationship between today and yesterday
from statsmodels.graphics.tsaplots import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

acf(bitcoin['dbprice'].dropna())
pacf(bitcoin['dbprice'].dropna())

# Plots for acf and pacf
plot_acf(bitcoin['dbprice'].dropna())
plot_pacf(bitcoin['dbprice'].dropna())


######## Step 10 #########

from statsmodels.tsa.arima_model import ARIMA

x = bitcoin['doil']
bigX = sm.add_constant(pd.concat((x,bitcoin['deuro'],bitcoin['dgold'],bitcoin['dsp']),1))[1:]
x = x[1:]
y = bitcoin['dbprice'][1:]
model10 = sm.OLS(y,bigX).fit()
y = bitcoin['dbprice'][1:]
model10 = sm.OLS(y,bigX).fit()
e = y-model10.predict()
model10.summary()
print(model10.aic)

# The best arima model with AIC value min is 
xarima = ARIMA(bitcoin['dbprice'],order=(0,1,0)).fit()  
print(xarima.aic) # AIC = -750.3871242322036

# we have to form a loop for finding p and q or orelse do manually
# All the models will not run. We should induce stationarity and choose different order
#xarima = ARIMA(bitcoin['bprice'],order=(1,1,1)).fit()
#xarima = ARIMA(bitcoin['bprice'],order=(1,1,2)).fit()
#xarima = ARIMA(bitcoin['bprice'],order=(1,1,3)).fit()
#xarima = ARIMA(bitcoin['bprice'],order=(2,1,1)).fit()
#xarima = ARIMA(bitcoin['bprice'],order=(2,1,2)).fit()
#xarima = ARIMA(bitcoin['bprice'],order=(3,1,1)).fit()
#xarima = ARIMA(bitcoin['bprice'],order=(3,1,1)).fit()


# Program runs without the below commented code. Confirm with Professor once.
#T = x.shape[0]
#mat = pd.DataFrame({'ndx':np.arange(T),'x':bitcoin['dbprice'].dropna().values,'e':e.values})
#mat.index = earima = ARIMA(mat['e'],order=(1,0,0)).fit()
#pd.to_datetime(mat['ndx'])
#mat['xo'] = xarima.resid
#mat['eo'] = earima.resid
#mat


######## Step 11 #########

bitcoin.index = bitcoin['Date']
xdata= pd.concat((bitcoin['bprice'],bitcoin['sp']),1)
modelarima = ARIMA(bitcoin['dbprice'],order=(0,1,0)).fit() 
modelarima.forecast()

# Other ways to forecast is to use sklearn with cross validation technique
from sklearn.model_selection import train_test_split
x = bigX
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 10)
model10 = sm.OLS(y_train,x_train).fit()
err = y_test - model10.predict(x_test)
mspe = (err**2).mean()
np.sqrt(mspe)

from sklearn.preprocessing import MinMaxScaler
mm_x = MinMaxScaler()
x_train = mm_x.fit_transform(x_train)
x_test = mm_x.fit_transform(x_test)

model11 = smf.OLS(y_train,sm.add_constant(x_train)).fit()
model11.summary()


######## Step 12 #########

from scipy import signal as sg
f, Pxx_den = sg.periodogram(bitcoin['bprice'], 10e3) # seasonality should be seen
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.semilogy(f, Pxx_den)
# Differencing Vairable
f, Pxx_den = sg.periodogram(bitcoin['dbprice'], 10e3) # should look like skyscrapers so no seasonality
# Still there is no seasonality confirm with professor
plt.semilogy(f, Pxx_den)


######## Step 13 #########

from statsmodels.tsa.api import VAR
bitcoin.index = bitcoin['Date']
xdata= pd.concat((bitcoin['bprice'],bitcoin['sp'],bitcoin['euro'],bitcoin['gold'],bitcoin['oil']),1)
model13 = VAR(xdata).fit(maxlags=3)
model13.summary()


######## Step 14 #########

# Forecasting using VAR model
model13.forecast(xdata.values,steps=30)
model13.plot()
