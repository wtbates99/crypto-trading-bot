# Bitcoin-Trader

## Purpose
This trading bot is meant to predict Bitcoins future price at whatever interval chosen, and place a trade if future price will be higher. Then creates a stoploss and take profit based off of future price and scans prices to close the trade based on those parameters.

I do **NOT** recommend live trading based off of predictions from this exact code; however, you can make the changes needed to make this deployable. This is *purely* an educational project, and data may be deceptive -- trade at your **OWN** risk **ALWAYS**!
## Overview 
The overall workflow to optimize machine learning to make and use price predictions is as follows:

1. Acquire historical fundamental data – these are the features or predictors
2. Acquire historical stock price data – this is will make up the dependent variable, or label (what we are trying to predict).
3. Preprocess data
4. Use a machine learning model to learn from the data
5. Acquire current fundamental data
6. Generate predictions from machine learning model using current data
7. Use a trading API to make trades based off of those predictions
8. Scan prices to hit take profit or stop loss points
  * Repeat steps 6-8


 

## General Notes
#### pip dependencies install: 
  * Navigate to file location and then run:
  * WINDOWS: py -m pip install -r requirements.txt 
  * LINUX: python -m pip install -r requirements.txt
  
#### Create an account with [Alpaca](https://app.alpaca.markets/paper/dashboard/overview)
  * This will allow you to obtain your API keys to input the trades
  * When you sign up, you will have access to a real money and a paper trading account
  
#### Notes for adding features
  * Can only grab info from any coin/USD if using crypto, or any stock in the S&P if using stocks
  * In the additional_data() function, you are able to grab other data you want used in the model
  
#### Changing model parameters or type
  * I have used mainly scikit-learn models for this project, although some test Tensorflow files are located in hte "TF" folder
  * I have used mainly regression models, but you can add a classifier to first check if the future price is higher than the current, then use a regression model to find the predicted price
  * I advise visiting the [scikit-learn documentation pages](https://scikit-learn.org/stable/supervised_learning.html) for more information on different models and parameters
