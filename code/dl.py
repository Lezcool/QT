# get today's date
import datetime
import yfinance as yf
import os
import pandas as pd
today = datetime.date.today()


company = ['NVDA','BABA','BILI','DIDIY','COIN','FFIE','META','SPCE','AAPL','NFLX','PDD','NET','NDX']
data_path = '/home/ec2-user/aria2-downloads/backtrader/QT/data'
for c in company:
    #download last day price and add to existing csv file
    # if file doesnt exit, then download
    file = os.path.join(data_path,f'{c}.csv')
    if False:
        df = pd.read_csv(file,index_col='Date',index='Date',parse_dates=True)
        last_day_df = pd.DataFrame(yf.Ticker(c).history().iloc[-1,:5]).T
        if last_day_df.index[-1].date() != str(df.index[-1]):
            new_df = pd.concat([df,last_day_df])
    else:
        if c == 'SPCE':
            fromdate = '2019-10-31'
        elif c == 'FFIE':
            fromdate = '2021-07-30'
        else:
            fromdate = '2017-01-01'
        df = yf.download(c, start=fromdate, end=None)
    df.to_csv(file)
