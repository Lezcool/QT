from __future__ import (absolute_import, division, print_function,unicode_literals)
import datetime 
import os
import sys  
import backtrader as bt
from prophet import Prophet
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import backtrader.analyzers as btanalyzers
import operator

from backtrader.utils.py3 import map
from backtrader import Analyzer, TimeFrame
from backtrader.mathsupport import average, standarddev
from backtrader.analyzers import AnnualReturn
import yaml

import logging
from tqdm import tqdm
# silence pyfolio warnings
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled=True
class TestStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
            elif order.issell():
                self.log('SELL EXECUTED, %.2f' % order.executed.price)

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            if self.dataclose[0] < self.dataclose[-1]:
                    # current close less than previous close

                    if self.dataclose[-1] < self.dataclose[-2]:
                        # previous close less than the previous close

                        # BUY, BUY, BUY!!! (with default parameters)
                        self.log('BUY CREATE, %.2f' % self.dataclose[0])

                        # Keep track of the created order to avoid a 2nd order
                        self.order = self.buy()

        else:

            # Already in the market ... we might sell
            if len(self) >= (self.bar_executed + 5):
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()

class SharpeRatio(Analyzer):
    params = (('timeframe', TimeFrame.Years), ('riskfreerate', 0.01),)

    def __init__(self):
        super(SharpeRatio, self).__init__()
        self.anret = AnnualReturn()

    def start(self):
        # Not needed ... but could be used
        pass

    def next(self):
        # Not needed ... but could be used
        pass

    def stop(self):
        retfree = [self.p.riskfreerate] * len(self.anret.rets)
        retavg = average(list(map(operator.sub, self.anret.rets, retfree)))
        retdev = standarddev(self.anret.rets)

        self.ratio = retavg / retdev

    def get_analysis(self):
        return dict(sharperatio=self.ratio)

class myStrategy(bt.Strategy):
    params = (
        ('maperiod', 15),
        ('printlog', False),
        ('args', None),
        ('beta',0.05), #
    )
    def log(self, txt, dt=None, doprint=False):
        ''' Logging function fot this strategy'''
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        
        df = pd.read_csv(args.data,index_col=0)
        self.ts = get_ts(df)

        #get final date in the data
        self.final_date = self.datas[0].datetime.date(0)

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add a MovingAverageSimple indicator
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.maperiod)
        # Add a RSI indicator
        self.rsi= bt.indicators.RSI_Safe(self.datas[0],period=14)
        # Add a MACD indicator
        self.macd = bt.indicators.MACDHisto(self.datas[0],period_me1=12,period_me2=26,period_signal=9)
        # Add a Stochastic indicator
        self.stoch = bt.indicators.Stochastic(self.datas[0],period=14,period_dfast=3,period_dslow=3)

        # recode values
        self.beta = self.params.beta
        self.highest = self.broker.get_cash()
        self.lowest = self.broker.get_cash()

        self.company = os.path.basename(args.data).split('.')[0]

        


    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))
        
    def ai_ind(self):
        #get current date
        date = self.datas[0].datetime.date(0)
        date = pd.to_datetime(date)
        train = self.ts.loc[self.ts.ds < date]

        m = Prophet(daily_seasonality=True, yearly_seasonality=True).fit(train)
        forecast = m.make_future_dataframe(periods=30, freq='D')
        pred = m.predict(forecast)
        yhat = pred.yhat.iloc[-1]
        close = self.dataclose[0]
        # print('stake'*10,stake)
        # print(f'{date},{round(cash,2)},yhat:[{round(yhat,2)},buy:{round(1.1*close,2)},sell:{round(0.9*close,2)}], close:{close},{stake}')
        # if yhat 10% higher than dataclose, buy

        if yhat > (1+self.beta)*close:
            return 'buy'
        # if yhat 10% lower than dataclose, sell
        elif yhat < (1-self.beta)*close:
            return 'sell'
        return 'hold'
    
    def sma_ind(self):
        if self.dataclose[0] > self.sma[0]:
            return 'buy'
        elif self.dataclose[0] < self.sma[0]:
            return 'sell'
        return 'hold'
    
    def rsi_ind(self):
        if self.rsi > 70:
            return 'sell'
        elif self.rsi< 30:
            return 'buy'
        return 'hold'
        
    def macd_ind(self):

        if self.macd[0] > 0 and self.macd[-1] <= 0:
            return 'buy'
        elif self.macd[0] < 0 and self.macd[-1] >= 0:
            # 当MACD柱状图从正值变为负值时，产生卖出信号
            return 'sell'
        return 'hold'
    
    def kdj_ind(self):
        # Stochastic strategy
        if self.stoch.percK[0] > 80:
            return 'sell'
        elif self.stoch.percK[0] < 20:
            return 'buy'
        return 'hold'
        # if self.stoch.percK[0] > self.stoch.percD[0] and self.stoch.percK[-1] <= self.stoch.percD[-1]:
        #     return 'buy'
        # elif self.stoch.percK[0] < self.stoch.percD[0] and self.stoch.percK[-1] >= self.stoch.percD[-1]:
        #     return 'sell'
        # return 'hold'

    def predict(self):
        if args.method == 'ai':
            return self.ai_ind()
        elif args.method == 'sma':
            return self.sma_ind()
        elif args.method == 'rsi':
            return self.rsi_ind()
        elif args.method == 'macd':
            return self.macd_ind()
        elif args.method == 'kdj':
            return self.kdj_ind()
        elif args.method == 'vote':
            action1 = self.ai_ind()
            action2 = self.sma_ind()
            action3 = self.rsi_ind()
            action4 = self.macd_ind()
            action5 = self.kdj_ind()
            if args.forcast: print(f'AI: {action1}, SMA {action2}, RSI: {action3}, MACD: {action4}, KDJ: {action5}')
            #count the number of buy and sell and hold
            # msa performs best
            buy_n, sell_n, hold_n = [[action1,action2,action4].count(i) for i in ['buy','sell','hold']]
            if buy_n >= 2:
                return 'buy'
            elif sell_n >= 2:
                return 'sell'
            else:
                return 'hold'
        else:
            return 'hold'


    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return
        
        #skip if is not final date
        # print(self.datas[0].datetime.date(0),self.final_date)
        if args.forcast and self.datas[0].datetime.date(0) < self.final_date:
            return
        
        cash = self.broker.get_cash()
        stake = self.broker.getposition(self.data).size
        close = self.dataclose[0]
        buyamount=int(cash*0.5/close)
        sellamount=int(stake*0.5)
        if stake ==1: sellamount = 1

        action = self.predict()

        if args.forcast: print(f'{self.datas[0].datetime.date(0)} Final Action: {action}')

        # Check if we are in the market
        # if not self.position:
        if action == 'buy':
            if buyamount > 0:
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy(size=buyamount)
        # else:
        elif action == 'sell':
            if sellamount > 0:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])
                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell(size=sellamount)

        # record highest and lowest portfolio value
        self.highest = max(self.highest, self.broker.getvalue())
        self.lowest = min(self.lowest, self.broker.getvalue())

    def stop(self):
        if args.forcast: return
        self.analyzers.mysharpe.stop()
        sharpe_ratio = self.analyzers.mysharpe.get_analysis()['sharperatio']
        if sharpe_ratio is None: sharpe_ratio = 0
        self.log('(MA Period %2d) (beta %2f) Ending Value %.2f Highest %.2f Lowest %.2f Sharp ratio %.2f' %
                 (self.params.maperiod, self.params.beta, self.broker.getvalue(),self.highest,self.lowest,sharpe_ratio), doprint=True)
        
        # append results to csv file
        with open(os.path.join(args.save_path,f'{args.method}_results.csv'), 'a') as f:
            f.write(f'{self.company},{self.params.maperiod},{self.params.beta},{self.broker.getvalue()},{self.highest},{self.lowest},{sharpe_ratio}\n')
        f.close()
        
    
# add arguement to the script
def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            'Backtrader test script'
        )
    )
    parser.add_argument('--data', default='/zhome/dc/1/174181/docs/QT/data/NVDA.csv',
                        required=False, help='Data to read in')
    parser.add_argument('--method', default='ai',help='ai, sma, rsi')
    parser.add_argument('--maperiod',type=int, default=10,help='sma period')
    parser.add_argument('--optimize',action='store_true' ,default=False,help='optimize sma')
    parser.add_argument('--beta',type=float, default=0.05 ,help='higher than beta %, buy')
    parser.add_argument('--folder_mode',action='store_true' ,default=False,help='run data set in folder')
    parser.add_argument('--plot',action='store_true' ,default=False,help='plot')
    parser.add_argument('--config',default='/zhome/dc/1/174181/docs/QT/code/setting.yml',help='maperiod config')
    parser.add_argument('--save_path',default='/zhome/dc/1/174181/docs/QT/results',help='save path')
    parser.add_argument('--forcast',action='store_true' ,default=False,help='forcast today')

    return parser.parse_args()


def get_ts(df):
    ts = df.reset_index()[['Date', 'Close']]
    ts.columns = ['ds', 'y']
    ts['ds'] = pd.to_datetime(ts['ds'])
    return ts

def main(args):
    # Create a cerebro entity
    cerebro = bt.Cerebro()
    df = pd.read_csv(args.data,index_col=0)
    ts = get_ts(df)
    sp = int(0.5*(len(ts)))
    fromdate,todate = ts.ds.iloc[sp], ts.ds.iloc[-1]
    # add one more day to make sure we have the last day
    todate = todate + datetime.timedelta(days=1)
    # Add a strategy
    if args.optimize:
        if args.method == 'sma':
            strats = cerebro.optstrategy(myStrategy, maperiod=range(10, 31))
        elif args.method == 'ai':
            strats = cerebro.optstrategy(myStrategy, beta=[0.05])
        elif args.method =='vote':
            strats = cerebro.optstrategy(myStrategy, maperiod=range(10, 31,2), beta=args.beta)
    else:
        cerebro.addstrategy(myStrategy,maperiod=args.maperiod,beta=args.beta)
    # 
    # modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    # datapath = os.path.join(modpath, '../../datas/orcl-1995-2014.txt')
    datapath = args.data
    filename = os.path.basename(datapath).split('.')[0]
    # Create a Data Feed
    data = bt.feeds.YahooFinanceCSVData(
        dataname=datapath,
        # Do not pass values before this date
        fromdate=fromdate,
        # Do not pass values before this date
        todate=todate,
        # Do not pass values after this date
        reverse=False)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)

    # Add a FixedSize sizer according to the stake
    # cerebro.addsizer(bt.sizers.FixedSize, stake=10)

    # Set the commission
    cerebro.broker.setcommission(commission=0.001)

    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='mysharpe')

    # Run over everything
    cerebro.run()
    # print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    if not args.optimize and args.plot:
        cerebro.plot()
        plt.savefig(os.path.join(args.save_path,f'{filename}_{args.method}.png'))


if __name__ == '__main__':
    args = parse_args()
    # create new csv file
    with open(os.path.join(args.save_path,f'{args.method}_results.csv'), 'w') as f:
        f.write('company,maperiod,beta,ending value,highest,lowest,sharp_ratio\n')
    f.close()

    file = args.config
    with open(file) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    file.close()

    if args.folder_mode:
        folder = args.data
        for file in tqdm(os.listdir(folder)):
            if file.endswith('.csv'):
                name = file.split('.')[0]
                try:
                    args.maperiod = params[name]
                except:
                    args.maperiod = 20
                args.data = os.path.join(folder,file)
                print('*'*10,name,'*'*10)
                main(args)

    else:
        main(args)