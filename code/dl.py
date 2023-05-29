# get today's date
import datetime
import yfinance as yf
import os
today = datetime.date.today()


company = ['NVDA','BABA','BILI','DIDIY','COIN','FFIE','META','SPCE','AAPL','NFLX','PDD','NET']

for c in company:
    if c == 'SPCE':
        fromdate = '2019-10-31'
    elif c == 'FFIE':
        fromdate = '2021-07-30'
    else:
        fromdate = '2017-01-01'
    df = yf.download(c, start=fromdate, end=today)
    df.to_csv(os.path.join('/zhome/dc/1/174181/docs/QT/data',f'{c}.csv'))
