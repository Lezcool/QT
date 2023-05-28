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

import logging
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

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add a MovingAverageSimple indicator
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod)
        
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
        
    def ai_desicion(self):
        #get current date
        date = self.datas[0].datetime.date(0)
        date = pd.to_datetime(date)
        train = self.ts.loc[self.ts.ds < date]

        m = Prophet(daily_seasonality=True, yearly_seasonality=True).fit(train)
        forecast = m.make_future_dataframe(periods=30, freq='D')
        pred = m.predict(forecast)
        yhat = pred.yhat.iloc[-1]
        cash = self.broker.get_cash()
        stake = self.broker.getposition(self.data).size
        close = self.dataclose[0]
        
        # print('stake'*10,stake)
        # print(f'{date},{round(cash,2)},yhat:[{round(yhat,2)},buy:{round(1.1*close,2)},sell:{round(0.9*close,2)}], close:{close},{stake}')
        # if yhat 10% higher than dataclose, buy

        if yhat > (1+self.beta)*close:
            amount=int(cash*0.5/close)
            if amount > 0:
                return 'buy', amount
        # if yhat 10% lower than dataclose, sell
        elif yhat < (1-self.beta)*close:
            amount=int(stake*0.5)
            if stake ==1: amount = 1
            if amount > 0:
                return 'sell', amount
        return 'hold', 0
    
    def sma_desicion(self):
        cash = self.broker.get_cash()
        stake = self.broker.getposition(self.data).size
        close = self.dataclose[0]
        if self.dataclose[0] > self.sma[0]:
            amount=int(cash*0.5/close)
            return 'buy', amount
        elif self.dataclose[0] < self.sma[0]:
            amount=int(stake*0.5)
            if stake ==1: amount = 1
            return 'sell', amount
        return 'hold', 0

    def predict(self):
        if args.method == 'ai':
            return self.ai_desicion()
        elif args.method == 'sma':
            return self.sma_desicion()

        elif args.method == 'vote':
            action1, amount1 = self.ai_desicion()
            action2, amount2 = self.sma_desicion()
            if action1 == action2:
                return action1, max(amount1, amount2)
            else:
                return 'hold', 0
        else:
            return 'hold', 0


    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        action, amount = self.predict()
        # Check if we are in the market
        # if not self.position:
        if action == 'buy':
            
            self.log('BUY CREATE, %.2f' % self.dataclose[0])
            # Keep track of the created order to avoid a 2nd order
            self.order = self.buy(size=amount)
        # else:
        elif action == 'sell':
            # SELL, SELL, SELL!!! (with all possible default parameters)
            self.log('SELL CREATE, %.2f' % self.dataclose[0])
            # Keep track of the created order to avoid a 2nd order
            self.order = self.sell(size=amount)

        # record highest and lowest portfolio value
        self.highest = max(self.highest, self.broker.getvalue())
        self.lowest = min(self.lowest, self.broker.getvalue())

    def stop(self):
        self.log('(MA Period %2d) (beta %2f) Ending Value %.2f Highes %.2f Lowest %.2f' %
                 (self.params.maperiod, self.params.beta, self.broker.getvalue(),self.highest,self.lowest), doprint=True)
        self.analyzers.mysharpe.stop()
        sharpe_ratio = self.analyzers.mysharpe.get_analysis()['sharperatio']
        # append results to csv file
        with open(f'{args.method}_results.csv', 'a') as f:
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
    parser.add_argument('--method', default='ai',help='ai or sma')
    parser.add_argument('--maperiod',type=int, default=10,help='sma period')
    parser.add_argument('--optimize',action='store_true' ,default=False,help='optimize sma')
    parser.add_argument('--beta',type=float, default=0.1 ,help='higher than beta %, buy')
    parser.add_argument('--folder_mode',action='store_true' ,default=False,help='run data set in folder')

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

    # Add a strategy
    if args.optimize:
        if args.method == 'sma':
            strats = cerebro.optstrategy(myStrategy, maperiod=range(10, 31))
        elif args.method == 'ai':
            strats = cerebro.optstrategy(myStrategy, beta=[0.05,0.1,0.15,0.2,0.25,0.3])
        elif args.method =='vote':
            strats = cerebro.optstrategy(myStrategy, maperiod=range(10, 31,2), beta=0.05)
    else:
        cerebro.addstrategy(myStrategy,maperiod=args.maperiod,beta=args.beta)
    # 
    # modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    # datapath = os.path.join(modpath, '../../datas/orcl-1995-2014.txt')
    datapath = args.data
    filename = os.path.basename(datapath)
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
    if not args.optimize:
        cerebro.plot()
        plt.savefig(os.path.join('/results',f'{filename}_{args.method}.png'))


if __name__ == '__main__':
    args = parse_args()
    # create new csv file
    with open(f'{args.method}_results.csv', 'w') as f:
        f.write('company,maperiod,beta,ending value,highest,lowest,sharp_ratio\n')
    f.close()
    if args.folder_mode:
        folder = args.data
        for file in os.listdir(folder):
            if file.endswith('.csv'):
                args.data = os.path.join(folder,file)
                print('*'*10,file,'*'*10)
                main(args)

    else:
        main(args)