#basic libraries
import os
import json
import time
from datetime import datetime
import csv
#stock data, plots, dataframes
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
#array math and stats
import numpy as np
import scipy
from scipy import stats
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
#machine learning models
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
#sentiment analysis
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#http requests and decoders
import requests as request
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
#warning remover
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#where to save results
full_starttime = datetime.now()
#trade placing libraries
from os.path import exists as file_exists
from tqdm import tqdm
from alpaca.trading import TradingClient
from alpaca.trading import GetAssetsRequest
from alpaca.trading import MarketOrderRequest
from alpaca.trading import OrderSide, TimeInForce
#alpaca information
API_KEY = "<input api key>"
SECRET_KEY = "<input secret key>"
APCA_API_BASE_URL = "https://paper-api.alpaca.markets"



#basic inputs
how_much_time = '2m'
#generate a new model even if one already exists (True = Yes, False = No)
if_model_already_exists_generate_new_model = True
#run model forever or just once (True = Forever, False = Once)
runforever = False


if how_much_time == '1m':
    period, interval, sleepytime, price_change_sleep = "7d", "1m", 1.2, 1.2
    one_period, two_period, three_period, four_period, five_period = 60, 120, 300, 600, 1440
elif how_much_time == '2m':
    period, interval, sleepytime, price_change_sleep = "60d", "2m", 1.2, 1.2
    one_period, two_period, three_period, four_period, five_period = 30, 60, 120, 300, 720
elif how_much_time == '5m':
    period, interval, sleepytime, price_change_sleep = "60d", "5m", 3, 1.5
    one_period, two_period, three_period, four_period, five_period = 12, 24, 60, 120, 288
elif how_much_time == '15m':
    period, interval, sleepytime, price_change_sleep = "60d", "15m", 3, 3
    one_period, two_period, three_period, four_period, five_period = 4, 8, 20, 40, 100
elif how_much_time == '1h':
    period, interval, sleepytime, price_change_sleep = "60d", "1h", 3, 3
    one_period, two_period, three_period, four_period, five_period = 2, 4, 10, 20, 50
elif how_much_time == '1d':
    period, interval, sleepytime, price_change_sleep = "max", "1d", 3, 3
    one_period, two_period, three_period, four_period, five_period = 2, 4, 10, 20, 30
elif how_much_time == '5d':
    period, interval, sleepytime, price_change_sleep = "max", "5d", 3, 3
    one_period, two_period, three_period, four_period, five_period = 1, 2, 3, 4, 5
else:
    print('INTERVAL NOT AVAILABLE')




#tickers
tickerlist = ['BTC-USD',
                ]
#   printtickerlist
print('--------------------------------------------\n' 'Tickers: ' + str(tickerlist))            
          
def data_collector():
    ticker = tickerlist[cnt].upper()
    stock = yf.Ticker(ticker)
    global stock_hist
    stock_hist = stock.history(period=period, interval=interval)

    ###moving data to find out difference in prices between two days###
    stock_prev = stock_hist.copy()
    stock_prev = stock_prev.shift(1)

    ###finding actual close###
    data = stock_hist[["Close"]]
    data = data.rename(columns = {'Close':'Actual_Close'})

    ###setup our target###
    data["Target"] = stock_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]

    ###join the data###
    predict = ["Close", "Volume", "Open", "High", "Low"]
    data_prev = data.join(stock_prev[predict]).iloc[0:]
    data_now = data.join(stock_hist[predict]).iloc[0:]

    ###adding in more data
    data_full = moredata(data_prev)
    data_today = moredata(data_now)
    close_data = moredata(data_prev)
    
    ###removing features
    data_full = data_remove(data_full)
    data_today = data_remove(data_today)

    return data_full, data_today, close_data, ticker
    
def moredata(data_alter):
    
    #adding in predicted ticker
    close_one = data_alter.rolling(one_period).mean()["Close"]
    close_two = data_alter.rolling(two_period).mean()["Close"]
    close_three = data_alter.rolling(three_period).mean()["Close"]
    close_four = data_alter.rolling(four_period).mean()["Close"]
    close_five = data_alter.rolling(five_period).mean()["Close"]
    x_trend = data_alter.shift(1).rolling(five_period).sum()["Target"]
    volume_one = data_alter.rolling(one_period).mean()["Volume"]
    volume_two = data_alter.rolling(two_period).mean()["Volume"]
    volume_three = data_alter.rolling(three_period).mean()["Volume"]
    volume_four = data_alter.rolling(four_period).mean()["Volume"]
    volume_five = data_alter.rolling(five_period).mean()["Volume"]
    data_alter["BTC_trend"] = x_trend
    data_alter["BTC_c_one"] = close_one
    data_alter["BTC_c_two"] = close_two
    data_alter["BTC_c_three"] = close_three
    data_alter["BTC_c_four"] = close_four
    data_alter["BTC_c_five"] = close_five
    data_alter["BTC_v_one"] = volume_one
    data_alter["BTC_v_two"] = volume_two
    data_alter["BTC_v_three"] = volume_three
    data_alter["BTC_v_four"] = volume_four
    data_alter["BTC_v_five"] = volume_five
    #adding in extra data
    def additional_data(data, tick, close_name, volume_name):
    
        tick_input = yf.Ticker(tick)
        base_input = tick_input.history(period=period, interval=interval)
        #close data
        c_input = base_input.rename(columns = {'Close':close_name})
        c_input = c_input[close_name]
        c_one = c_input.rolling(one_period).mean() 
        c_two = c_input.rolling(two_period).mean()
        c_three = c_input.rolling(three_period).mean()
        c_four = c_input.rolling(four_period).mean()
        c_five = c_input.rolling(five_period).mean()
            #add the close data
        data[tick + '_c_one'] = c_one
        data[tick + '_c_two'] = c_two
        data[tick + '_c_three'] = c_three
        data[tick + '_c_four'] = c_four
        data[tick + '_c_five'] = c_five
        data = data.join(c_input).iloc[1:]
        #volume data
        v_input = base_input.rename(columns ={'Volume':volume_name})
        v_input = v_input[volume_name]
        v_one = v_input.rolling(one_period).mean() 
        v_two = v_input.rolling(two_period).mean()
        v_three = v_input.rolling(three_period).mean()
        v_four = v_input.rolling(four_period).mean()
        v_five = v_input.rolling(five_period).mean()
            #add the volume data
        data[tick + '_v_one'] = v_one
        data[tick + '_v_two'] = v_two
        data[tick + '_v_three'] = v_three
        data[tick + '_v_four'] = v_four
        data[tick + '_v_five'] = v_five
        data = data.join(v_input).iloc[1:]

        return data

    #LTC 
    data_alter = additional_data(data_alter, 'LTC-USD', "LTC_Close", "LTC_Volume")
    
    #ETC
    data_alter = additional_data(data_alter, 'ETC-USD', "ETC_Close", "ETC_Volume")
    #AAPL 
    #data_alter = additional_data(data_alter, 'AAPL', "AAPL_Close", "AAPL_Volume")
    data_alter = data_alter.dropna(how='any')
    return data_alter


def data_remove(x):
    x = x.drop([
                ], axis=1)
    return x
    
def model_creator(data_full, ticker, model_creator_starttime, cnt):
    print('\nGenerating model for: ' + ticker)
    #scale the data
    scaler = preprocessing.MinMaxScaler()

    #model and data
    y = data_full['Actual_Close']
    X = data_full.drop(['Actual_Close'], axis=1)
    
    #split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=23, shuffle=False)

    #select a model
    model = ExtraTreesRegressor(criterion='poisson', bootstrap=True, oob_score=True, random_state=23)

    
    #parameters to be tuned
    parameters = {
        'n_estimators' : sp_randInt(50, 750),
        'min_samples_split'  : sp_randInt(1, 4),
        'min_samples_leaf' : sp_randInt(1,3),


                    
                    }
   
    print("Optimization for:", ticker)  
    model = RandomizedSearchCV(estimator=model, param_distributions = parameters,
        refit=True, cv = 5, n_iter = 20, verbose = 2, n_jobs=-1, random_state=23)

    #make training sets
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


    print("Model fit for :", ticker)
    print("Optimization complete for :", ticker + "\n----------------")
    #prediction and model statistics
    y_pred = model.predict(X_test)
    print("Model statistics for: " + ticker + "\n========")
    print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', mean_squared_error(y_test, y_pred))

    ###SAVE THE MODEL###
    with open('model_' + ticker + '.pkl', 'wb') as f:
        pickle.dump(model,f)
    print("----------------\nModel for: ", ticker, "has been saved to file location\n----------------")
    print("----------------\nRuntime for:", ticker, datetime.now()-model_creator_starttime, "\n--------------------------------------------")

def model_run():
    global cnt
    cnt = 0
    for i in tickerlist:
        model_creator_starttime = datetime.now()
        if if_model_already_exists_generate_new_model == False:
            if file_exists('model_' + i + '.pkl'):
                print('--------------------------------------------\nModel for ' + i +  ' already exists...\n--------------------------------------------')
                cnt = cnt + 1
            else:
                data_full, data_today, close_data, ticker = data_collector() 
                model_creator(data_full,ticker, model_creator_starttime, cnt)
                cnt = cnt + 1
        else:
            data_full, data_today, close_data, ticker = data_collector()
            model_creator(data_full, ticker, model_creator_starttime, cnt)
            cnt = cnt + 1

    global model_creator_runtime
    model_creator_runtime = (datetime.now() - model_creator_starttime, cnt)

def model_predictor():
    global cnt
    cnt = 0
    model_predictor_starttime = datetime.now()
    final_df = pd.DataFrame(columns = 
        ['Date and hour',
        'Ticker', 
        'Last_Pred_Close', 
        'Last_Actual_Close', 
        'Now_Pred_Close', 
        'Now_Actual_Close', 
        'Difference', 
        'Next_Pred_Close' 
        ])

    while cnt < len(tickerlist):
        data, data_today, close_data, ticker = data_collector()

        ###open model###
        with open('model_' + ticker + '.pkl', 'rb') as f:
            model = pickle.load(f)

        #todays predictions
        data_full = data.drop(['Actual_Close'], axis=1)
        y_pred = model.predict(data_full.tail(2))
        y_pred_fixed = np.delete(y_pred, 1)
        y_pred_tmrw = model.predict(data_full.tail(1))
        yest_close_fixed = close_data.tail(1)["Close"]

        #tomorrows predictions
        data_today = data_today.drop(['Actual_Close'], axis=1)
        y_pred_tmrw_tmrw = model.predict(data_today.tail(1))


        final_df = final_df.append(
        {'Date and hour' :     time.strftime("%m_%d_%Y_%H"),
        'Ticker' : ticker,
        'Last_Pred_Close' : round(float(y_pred_fixed), 2),
        'Last_Actual_Close' : round(float(yest_close_fixed), 2),
        'Now_Pred_Close' : round(float(y_pred_tmrw), 2),
        'Now_Actual_Close' : round(float(data.tail(1)["Actual_Close"]), 2),
        'Difference' : round(float(float(y_pred_tmrw) - data.tail(1)["Actual_Close"]), 2),
        'Next_Pred_Close' : round(float(y_pred_tmrw_tmrw), 2)},
        ignore_index = True)
        print(ticker, "close predicted \n--------------------------------------------")
        cnt = cnt + 1
        
    return final_df, model_predictor_starttime
    
def sentiment(ticker):
    ###SENTIMENT ANALYSIS###
    web_url = 'https://finviz.com/quote.ashx?t='
    news_tables = {}
    tickers = ticker
    url = web_url + ticker
    req = Request(url=url,headers={"User-Agent": "Chrome"}) 
    response = urlopen(req)    
    html = BeautifulSoup(response,"html.parser")
    news_table = html.find(id='news-table')
    news_tables[ticker.upper()] = news_table 
    snews = news_tables[ticker]
    snews_tr = snews.findAll('tr')
    for x, table_row in enumerate(snews_tr):
        a_text = table_row.a.text
        td_text = table_row.td.text
        if x == 3:
            break
            
    news_list = []
    for file_name, news_table in news_tables.items():
        for i in news_table.findAll('tr'):                
            try:
                text = i.a.get_text() 
            except AttributeError:
                print('')      
                
            datex_scrape = i.td.text.split()
            if len(datex_scrape) == 1:
                timex = datex_scrape[0]                  
            else:
                datex = datex_scrape[0]
                timex = datex_scrape[1]

            tick = file_name.split('_')[0]                
            news_list.append([tick, datex, timex, text])          
    vader = SentimentIntensityAnalyzer()
    columns = ['ticker', 'date', 'time', 'headline']
    news_df = pd.DataFrame(news_list, columns=columns)
    scores = news_df['headline'].apply(vader.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)
    news_df = news_df.join(scores_df, rsuffix='_right')
    news_df['date'] = pd.to_datetime(news_df.date).dt.date
    mean_scores = news_df.groupby(['ticker','date']).mean()
    mean_scores = mean_scores.unstack()
    mean_scores = mean_scores.xs('compound', axis="columns").transpose()
    mean_scores = mean_scores.tail(3)
    mean_scores = mean_scores.mean()
    averagescore = float(mean_scores)
    return averagescore

def predictor_and_printer():
    now_ = time.strftime("%m_%d_%Y_%H")
    final_df, model_predictor_starttime = model_predictor()
    print('\n--------------------------------------------\nPredicted Prices')
    print(final_df)

    global model_predictor_runtime 
    model_predictor_runtime = (datetime.now()-model_predictor_starttime) 
    return final_df

def trade_placer(trade_data):
    trade_data_fixed = trade_data[['Ticker',
                                  'Now_Pred_Close',
                                  'Now_Actual_Close',
                                  'Next_Pred_Close']].copy()
         
    #trading bot
    ###########

    difference_list = []
    diff_count = 0
    for stocks in range(len(trade_data_fixed)):
        tmrwclose = trade_data_fixed.iloc[diff_count, 3]
        tdyclose = trade_data_fixed.iloc[diff_count, 2]
        differences = tmrwclose - tdyclose
        difference_list.append(differences)
        diff_count = diff_count + 1


    trade_data_fixed.insert(4, "Pred Earn", difference_list, True)
    trade_data_fixed = trade_data_fixed.sort_values(by=['Pred Earn'], ascending=False)

    ###show stocks that passed through the requirement
    print('\n--------------------------------------------\nSorted by Predicted Earnings\n', trade_data_fixed)

    if differences >= 0:
    
        # Get our account information.
        HEADERS = {'APCA-API-KEY-ID': API_KEY, 'APCA-API-SECRET-KEY': SECRET_KEY}
        trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        account = trading_client.get_account()


        # Check if our account is restricted from trading.
        if account.trading_blocked:
            print('Account is currently restricted from trading.')



        print('\n----------------------------------------------------------------------------------------\n----------------------------------------------------------------------------------------\nLOGGED INTO TRADING ACCOUNT\n----------------------------------------------------------------------------------------\n----------------------------------------------------------------------------------------\n')
        # Check how much money we can use to open new positions.
        print('----------------\nBuying Power: ')
        print('${} is available as cash.'.format(account.cash))
        print('${} is available as buying power.'.format(account.buying_power))
        print('${} is held as equity.'.format(account.equity))
        #setup trading parameters for quantity and ticker
        bitcoin_p = bitcoin_price()
        bitcoin_p = str(bitcoin_p)
        cash_for_trades = float(account.cash) * 1
        cash_per_stock = cash_for_trades * 0.15
        buy_price = trade_data_fixed.iloc[0, 2]
        buy_tick = trade_data_fixed.iloc[0, 0]
        buy_tick = buy_tick.replace("-", "/")
        quantity = round(cash_per_stock / buy_price, 4)
        profit = trade_data_fixed.iloc[0, 3]
        take_profit_price = round(profit - .01)
        stop_loss_price = round(take_profit_price - buy_price, 2)
        stop_loss_price = round(buy_price - stop_loss_price)
        cashleftover = quantity * buy_price
        cashleftover = round(float(account.cash) - cashleftover, 2)
        print('\n----------------\nSENDING TRADE:', buy_tick, 'FOR', quantity, 'SHARES AT $', buy_price, '||| Total $$$', round(quantity*buy_price, 2))
        print('STOP LOSS PRICE: $', stop_loss_price)
        print('TAKE PROFIT PRICE: $', take_profit_price)
        print('\nCASH LEFTOVER: $', cashleftover, '\n')
        # Place the bracket order
        data = {
                "symbol": buy_tick,
                "qty": quantity,
                "side": "buy",
                "type": "market",
                "time_in_force": "gtc",

            }

        response = request.post(f"{APCA_API_BASE_URL}/v2/orders", headers=HEADERS, json=data)

        if response.status_code == 200:
            print('\n--------------------------------------------')
            print('TRADE SUCCESFULLY SENT:', buy_tick, 'FOR', quantity, 'SHARES AT $', buy_price, '||| Total $$$', round(quantity*buy_price, 2))
            print('STOP LOSS PRICE: $', stop_loss_price)
            print('TAKE PROFIT PRICE: $', take_profit_price)
            print('CASH LEFTOVER: $', cashleftover)
            print('\n--------------------------------------------')
            print('LET THE ALGO BE BLESSED BY LOML')
            print('\n--------------------------------------------')
            
            scan_tick = buy_tick.replace("/","-")
            scan = pricescanner(scan_tick) 
            loop_number = 0
            
            #Stoploss and Takeprofit 
            while scan >= stop_loss_price and scan <= take_profit_price:
                loop_number = loop_number + 1
                scan = pricescanner(scan_tick)
                #Stoploss
                if scan <= stop_loss_price:
                    print('\n----------------\nPLACING SELL AT A LOSS')
                    #Setup the bracket order
                    data = {
                        "symbol": buy_tick,
                        "qty": quantity,
                        "side": "sell",
                        "type": "market",
                        "time_in_force": "gtc",
                    }
                    #Place sell for stoploss
                    response = request.post(f"{APCA_API_BASE_URL}/v2/orders", headers=HEADERS, json=data)
                    #Sold for a stoploss
                    if response.status_code == 200:
                        print('\n--------------------------------------------')
                        print('SOLD FOR A LOSS')
                    #Do another scan
                    else:
                        print('\n--------------------------------------------')

                #Take profit
                if scan >= take_profit_price:
                    print('\n----------------\nPLACING SELL AT A GAIN!!!')
                    #Setup the bracket order
                    data = {
                        "symbol": buy_tick,
                        "qty": quantity,
                        "side": "sell",
                        "type": "market",
                        "time_in_force": "gtc",
                         }
                    #Place sell for take profit
                    response = request.post(f"{APCA_API_BASE_URL}/v2/orders", headers=HEADERS, json=data)
                    #Sold a a take profit
                    if response.status_code == 200:
                        print('\n--------------------------------------------')
                        print('SOLD FOR A GAIN!!!')
                    #Do another scan
                    else:
                        print('\n--------------------------------------------')
                #Loop and price data
                print('When the phone rings... will you answer? Loop#', loop_number)
                print('       Stop loss price of:', stop_loss_price)
                print('       Current Price of:', scan)
                print('       Take profit price of:', take_profit_price, '\n\n=============\n')
                #Waiting to check current price against stoploss/take profit
                for i in tqdm (range (100), desc="Waiting to check for price change...", ascii=False, ncols=75, colour='GREEN'):
                    time.sleep(price_change_sleep)
        #Error placing order
        else:
            print('\n--------------------------------------------')
            print(f"ERROR PLACING BRACKET ORDER: {response.content}")
            print('\n--------------------------------------------')
    #Predicted future close is lower than current price
    else:
        print('NO MONEY TO BE MADE')
    #Set a timer to repredict price
    for i in tqdm (range (100), desc="Waiting for a positive predicted earning...", ascii=False, ncols=75, colour='GREEN'):
        time.sleep(sleepytime)



#---------------------------------------------------------
#Unused - trying to get real time coin data
#---------------------------------------------------------

def pricescanner(tocheck):
    scan_stock = yf.Ticker(tocheck)
    scan_stock_hist = scan_stock.history(period='1d')
    scan_close = scan_stock_hist['Close']
    scan_price = round(float(scan_close), 2) 
    return scan_price

def bitcoin_price():
    url = 'https://www.google.com/finance/quote/BTC-usd'

    # send a GET request to the website and get the HTML content
    response = request.get(url)
    html = response.content
    soup = BeautifulSoup(html, 'lxml')
    divs = soup.find_all('div', {'class': 'YMlKec fxKbKc'})
    divs = str(divs)
    divs = divs.split(">")
    price = (divs[1])
    price = price.split("<")
    price = price[0]
    return price

#---------------------------------------------------------------------
#start the model
#---------------------------------------------------------------------
model_run()
deet = predictor_and_printer()
deet2 = trade_placer(deet)

while runforever == True:
    yeet = predictor_and_printer()
    yeet2 = trade_placer(yeet)

print("Predictor Runtime: ", model_predictor_runtime)  
print("Full Runtime: ", datetime.now()-full_starttime, "\n--------------------------------------------")
