######### Practice Set 1  ###########
#### Varun Varadaraj Kandasamy ######
########### vxk161230  #############

# Packages that are responsible for connecting the database and extracting the information
library(data.table)
library(DBI)
library(RSQLite)
library(lmtest) # More advanced hypothesis testing tools
library(tseries) # Time series package


######## Step 1 #########
# Below is the link for bitcoin price
# https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20140101&end=20180619


######## Step 2 #########
# From Fred Euro, Oil, Gold, S&P500 date are collected and merged using Excel


######## Step 3 #########
# Bitcoin date
data1 <- read.csv('bitcoin.csv')
# Euro, Oil, Gold, S&P500 data
data2 <- read.csv('fred.csv')
# Merging both the datasets using column date
# Inner Join
bitcoin <- merge(data1, data2, all=FALSE)
bitcoin$Date <- as.Date(bitcoin$Date)


######## Step 4 #########
# Plot Series in R
library(ggplot2)
tmp <- ggplot(bitcoin, aes(x = Date,y = bprice)) +geom_line()
tmp
tmp <- ggplot(bitcoin, aes(x = Date,y = sp)) +geom_line()
tmp
tmp <- ggplot(bitcoin, aes(x = Date,y = oil)) +geom_line()
tmp
tmp <- ggplot(bitcoin, aes(x = Date,y = oil)) +geom_line()
tmp
tmp <- ggplot(bitcoin, aes(x = Date,y = gold)) +geom_line()
tmp
# Time Series Graph
ts.plot(bitcoin$Date)
ts.plot(bitcoin$bprice)
ts.plot(bitcoin$sp)
ts.plot(bitcoin$oil)
ts.plot(bitcoin$euro)
ts.plot(bitcoin$gold)


######## Step 5 #########
# here for Euro alone we no need to use log, but even if you used its not a big problem.
model5 <- lm(log(bprice)~log(sp)+log(oil)+log(euro)+log(gold), data = bitcoin)
model5
summary(model5)
# Apart from gold, all the other variables are significant so its total bullshit relationship


######## Step 6 #########
# Always set the stationary to Null hypothesis 
# H0 : stationary
# H1 : Non stationary

# we have to always check for larger pvalue (0.1), 
# so that we can prove its insignificant and not reject null hypothesis

# bprice
kpss.test(log(bitcoin$bprice), null = 'Trend')
kpss.test(log(bitcoin$bprice), null = 'Level') 
# Taking Differencing
kpss.test(diff(log(bitcoin$bprice)), null = 'Trend') # p-value 0.1
kpss.test(diff(log(bitcoin$bprice)), null = 'Level') # p-value 0.1
# level and Trend stationary proved with first differencing.

# sp
kpss.test(log(bitcoin$sp), null = 'Trend')
kpss.test(log(bitcoin$sp), null = 'Level')
# Taking Differencing
kpss.test(diff(log(bitcoin$sp)), null = 'Trend') # p-value 0.1
kpss.test(diff(log(bitcoin$sp)), null = 'Level') # p-value 0.1
# Trend and Level Stationary proved with 1st differencing

#oil
kpss.test(log(bitcoin$oil), null = 'Trend')
kpss.test(log(bitcoin$oil), null = 'Level')
# Taking Differencing
kpss.test(diff(log(bitcoin$oil)), null = 'Trend') # p-value 0.1
kpss.test(diff(log(bitcoin$oil)), null = 'Trend') # p-value 0.1
# Oil has been in Trend and level stationary at first differncing 

#euro
kpss.test(log(bitcoin$euro), null = 'Trend')
kpss.test(log(bitcoin$euro), null = 'Level')
# Taking Differencing
kpss.test(diff(log(bitcoin$euro)), null = 'Trend')# p-value 0.1
kpss.test(diff(log(bitcoin$euro)), null = 'Level')# p-value 0.1
# Euro has been in Level and Trend Stationary at first differncing.

#gold
kpss.test(log(bitcoin$gold), null = 'Trend')
kpss.test(log(bitcoin$gold), null = 'Level')
# Taking Differencing
kpss.test(diff(log(bitcoin$gold)), null = 'Trend') # p-value 0.1
kpss.test(diff(log(bitcoin$gold)), null = 'Level') # p-value 0.1
# Trend-stationary is proved with 1st differencing.


######## Step 7 #########
# bprice, gold, sp, oil, euro (1st diffrencing)
model7 <- lm(diff(log(bprice))~diff(log(sp))+diff(log(oil))+diff(log(euro))+diff(log(gold))
             , data = bitcoin)
head(bitcoin)
coef(model7)
?log
model7
summary(model7)
# After taking differences all the other variables become insignificant.


######## Step 8 #########
# bitcoin$Date <- as.Date(bitcoin$Date, format="%m/%d/%Y")
bitcoin$Date <- as.Date(bitcoin$Date)
# bitcoin <- filter(bitcoin, Date >= "2017-01-01")
bitcoin <- subset(bitcoin, Date >= "2017-01-01")
tmp <- ggplot(bitcoin, aes(x = Date,y = bprice)) +geom_line()
tmp

######## Step 9 #########
#acf gives the relationship between today and yesterday
acf(diff(log(bitcoin$bprice)))
pacf(diff(log(bitcoin$bprice)))


######## Step 10 #########
outp <- matrix(0L,7^2,3)
# ?matrix
row <- 1
# The i and j value are set from 0 to 6 because, from acf and pacf model
# the peak value for me is in between 0 and 6
for(i in 0:6){
  for(j in 0:6){
    aic <- AIC(arima(log(bitcoin$bprice),c(i,1,j)))
    outp[row,] <- c(i,j,aic)
    row <- row + 1
  }
}
order(outp[,3])
# The below output will give you the p,q value for which the AIC is minimum. 
# Choose that value for the rest of models.
outp[1,]
# p = 0 and q = 0
AIC(arima(log(bitcoin$bprice),c(0,1,0)))
# other way to calculate is to do manually without for loop
# AIC(arima(log(bitcoin$bprice),c(0,1,0)))
# AIC(arima(log(bitcoin$bprice),c(1,1,1)))
# AIC(arima(log(bitcoin$bprice),c(2,1,2)))
# AIC(arima(log(bitcoin$bprice),c(2,1,4)))
# AIC(arima(log(bitcoin$bprice),c(3,1,2)))
# AIC(arima(log(bitcoin$bprice),c(5,1,1)))


######## Step 11 #########
# ?arima
library(forecast)
model11 <- stats::arima(log(bitcoin$bprice),c(0,1,0))
# The values at the last are the p,d,q. In general case we do not know these values
steps <- 60
future <- forecast(model11,h=steps)
plot(future)


######## Step 12 #########
library(TSA)
# periodograms gives us the seasonality in the data.
# Our duty is to find the seasonality and remove it from the model.
# Frequency is the 1/number of periods.
periodogram(log(bitcoin$bprice))
# No seasonality 
periodogram(diff(log(bitcoin$bprice)))
# It looks like skyscrapers, so no seasonality changes.


######## Step 13 #########
# stationarity-transformed model
# model13 <- stats::arima(log(bitcoin$bprice),c(0,1,2),seasonal=list(order=c(1,0,1),period=5))
# The above commented model is the model with seasonal data, but our model has no seasonality
model13 <- stats::arima(log(bitcoin$bprice),c(0,1,0),seasonal=list(order=c(0,0,0),period=5))
model13$residuals
periodogram(model13$residuals)
# introduce a new funtion called weekdays, for creating dummy variables using bitcoin$date
bitcoin$weekday <- weekdays(bitcoin$Date)
bitcoin$weekday <- as.factor(bitcoin$weekday)
model135 <- lm(diff(log(bprice))~weekday[2:349],data=bitcoin)
# sum((model13$residuals-model135$residuals[2:839])^2)
periodogram(model135$residuals)
# There is not much difference between model13 and model 135, so there is no seasonality.


######## Step 14 #########
library(vars)
# VAR - vector auoto integration
diff_jp <- function(x){
  n <- nrow(x)
  return(x[2:n,]-x[1:n-1,])
}
x <- bitcoin %>% dplyr::select(bprice, sp, oil, euro, gold) %>% log %>% diff_jp
VAR(x,p=1,type="both") %>% AIC
# returns the first lag
VAR(x,p=2,type="both") %>% AIC
# returns the second lag
VAR(x,p=3,type="both") %>% AIC
# returns the third lag
# Creating a model with lag 3 is best since its AIC value is minimum compared to others.
model14 <- VAR(x,p=3,type="both")
summary(model14)
# To below is addtional program for finding how many days we have missed in an year
# nrow(subset(bitcoin, Date>"2017-01-01"&Date<"2018-01-01"))
# 365-240
# 125/2

######## Step 15 #########
model15 <- VAR(x,p=2,type="both")
summary(model15)
steps <- 60
?vars::predict
future15 <- predict(model15,n.ahead=steps)
future15



