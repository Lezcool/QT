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
# import backtrader.analyzers as btanalyzers
import operator

from backtrader.utils.py3 import map
from backtrader import Analyzer, TimeFrame
from backtrader.mathsupport import average, standarddev
from backtrader.analyzers import AnnualReturn
import yaml
import yfinance as yf

import logging
from tqdm import tqdm
# silence pyfolio warnings
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled=True

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
class ChandelierExit(bt.Indicator):

    ''' https://corporatefinanceinstitute.com/resources/knowledge/trading-investing/chandelier-exit/ '''

    lines = ('long', 'short')
    params = (('period', 22), ('multip', 3),)

    plotinfo = dict(subplot=False)

    def __init__(self):
        highest = bt.ind.Highest(self.data.high, period=self.p.period)
        lowest = bt.ind.Lowest(self.data.low, period=self.p.period)
        atr = self.p.multip * bt.ind.ATR(self.data, period=self.p.period)
        self.lines.long = highest - atr
        self.lines.short = lowest + atr

class myStrategy(bt.Strategy):
    params = (
        ('sma_period_short', 10),  # 较短SMA的计算周期
        ('sma_period_long', 20),
        ('printlog', False),
        ('args', None),
        ('beta',0.2), #
        ('n', 9),       # MACD快线的周期
        ('m', 5),       # 平仓时的K线数量
        ('atr_multiplier', 0.5),    # 最高/低价与ATR的乘数
        ('whichyhat','yhat'),
    )
    def log(self, txt, dt=None, doprint=False):
        ''' Logging function fot this strategy'''
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.company = os.path.basename(args.data).split('.')[0]
        
        df = pd.read_csv(args.data,index_col=0)
        self.ts = get_ts(df)
        try:
            self.ai_df = pd.read_csv(os.path.join(os.path.dirname(args.data),f'ai/{self.company}.csv'),parse_dates=True,header=None)
            self.ai_df.columns = ['date','yhat','yhat_lower','yhat_upper']
            self.ai_df.set_index('date',inplace=True)
            self.ai_df.index = pd.to_datetime(self.ai_df.index)
            #del duplicate rows
            self.ai_df = self.ai_df[~self.ai_df.index.duplicated(keep='first')]
        except:
            print('No AI data found')
            self.ai_df = None
        #get final date in the data
        self.final_date = self.datas[0].datetime.date(0)

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add a MovingAverageSimple indicator
        self.sma_short = bt.indicators.SimpleMovingAverage(self.data, period=self.params.sma_period_short)
        self.sma_long = bt.indicators.SimpleMovingAverage(self.data, period=self.params.sma_period_long)


        # Add a RSI indicator
        self.rsi= bt.indicators.RSI_Safe(self.datas[0],period=14)
        # Add a MACD indicator
        self.macd = bt.indicators.MACDHisto(self.datas[0],period_me1=12,period_me2=26,period_signal=9)
        # Add a Stochastic indicator
        self.stoch = bt.indicators.Stochastic(self.datas[0],period=14,period_dfast=3,period_dslow=3)
        # Add a TrendModel
        self.atr = bt.indicators.ATR(self.data)
        self.highs = bt.indicators.Highest(self.data.high, period=self.params.n)
        self.lows = bt.indicators.Lowest(self.data.low, period=self.params.n)
        # Add a ChandelierExit indicator
        self.ce = ChandelierExit(self.data, period=22, multip=3)
        

        # recode values
        self.beta = self.params.beta
        self.highest = self.broker.get_cash()
        self.lowest = self.broker.get_cash()
        self.sellcriteria = self.broker.get_cash()
        self.whichyhat = self.params.whichyhat

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
        close = self.dataclose[0]
        def calc_ai_y(date):
            train = self.ts.loc[self.ts.ds < date]
            m = Prophet(daily_seasonality=True, yearly_seasonality=True).fit(train)
            forecast = m.make_future_dataframe(periods=30, freq='D')
            pred = m.predict(forecast)
            yhat = pred.yhat.iloc[-1]
            yhat_lower = pred.yhat_lower.iloc[-1]
            yhat_upper = pred.yhat_upper.iloc[-1]
            
            # print('stake'*10,stake)
            # print(f'{date},{round(cash,2)},yhat:[{round(yhat,2)},buy:{round(1.1*close,2)},sell:{round(0.9*close,2)}], close:{close},{stake}')

            # recode values if on forcast mode and not last day of data
            if (self.params.forcast == True and date < self.final_date) or self.params.forcast == False:
                with open(f'/home/lez/Documents/QT/QT/data/ai/{self.company}.csv','a') as f:
                    f.write(f'{date},{yhat},{yhat_lower},{yhat_upper} \n')
                f.close()
            return yhat,yhat_lower,yhat_upper

        if self.ai_df is not None:
            try:
                date = pd.to_datetime(date)
                # print(date)
                # print(self.ai_df.loc[date,'yhat_upper'])
                yhat,yhat_lower,yhat_upper = self.ai_df.loc[date,'yhat'],self.ai_df.loc[date,'yhat_lower'],self.ai_df.loc[date,'yhat_upper']
                if isinstance(yhat,pd.core.series.Series): 
                    print('y_hat more than 1')
                    sys.exit()
            except:
                print(f'fail to to find {date}')
                yhat,yhat_lower,yhat_upper = calc_ai_y(date)
        else:
            yhat,yhat_lower,yhat_upper = calc_ai_y(date)
        # print(len(yhat))
        if self.whichyhat == 'yhat':
            y = yhat
        elif self.whichyhat == 'yhat_lower':
            y = yhat_lower
        elif self.whichyhat == 'yhat_upper':
            y = yhat_upper
        
        if y > (1+self.beta)*close:
            action = 'buy'           
        # if yhat 10% lower than dataclose, sell
        elif y < (1-self.beta)*close:
            action = 'sell'
        else:
            action = 'hold'

        return action
    
    def sma_ind(self):
        if self.sma_short[0] > self.sma_long[0] and self.sma_short[-1] <= self.sma_long[-1]:
            # 短期SMA上穿长期SMA，产生买入信号
            return 'buy'
        elif self.sma_short[0] < self.sma_long[0] and self.sma_short[-1] >= self.sma_long[-1]:
            # 短期SMA下穿长期SMA，产生卖出信号
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

    def TrendModel_ind(self):
        if self.macd.macd[0] > self.macd.signal[0] and self.macd.macd[-1] <= self.macd.signal[-1]:
            if self.data.close[0] > (self.highs[-1] + self.params.atr_multiplier * self.atr[-1]):
                return 'buy'
        
        if self.macd.macd[0] < self.macd.signal[0] and self.macd.macd[-1] >= self.macd.signal[-1]:
            if self.data.close[0] < (self.lows[-1] - self.params.atr_multiplier * self.atr[-1]):
                return 'sell'
            
        return 'hold'
    
    def ce_ind(self):
        # ChandelierExit indicator
        if self.ce.long[0] > self.data.close[0]:
            return 'sell'
        elif self.ce.short[0] < self.data.close[0]:
            return 'buy'
        return 'hold'

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
        elif args.method == 'trend':
            return self.TrendModel_ind()
        elif args.method == 'ce':
            return self.ce_ind()
        elif args.method == 'vote':
            action_ai = self.ai_ind()
            action_sma = self.sma_ind()
            action3 = self.rsi_ind()
            action_macd = self.macd_ind()
            action_kdj = self.kdj_ind()
            action_trend = self.TrendModel_ind()
            action_ce = self.ce_ind()
            if args.forcast: print(f'Price:{round(self.dataclose[0],2)}, AI+: {action_ai}, SMA+: {action_sma}, Trend+: {action_trend}, MACD+: {action_macd}, KDJ-: {action_kdj}')
            #count the number of buy and sell and hold
            # msa performs best
            if action_trend != 'hold':
                return action_trend
            elif action_ce == 'sell':
                return 'sell'
            else:
                buy_n, sell_n, hold_n = [[action_ai,action_sma,action_macd].count(i) for i in ['buy','sell','hold']]
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
        # if args.forcast and self.datas[0].datetime.date(0) < self.final_date:
        #     return
        
        cash = self.broker.get_cash()
        stake = self.broker.getposition(self.data).size
        close = self.dataclose[0]
        buyamount=int(cash*0.5/close)
        sellamount=int(stake*0.5)
        if stake ==1: sellamount = 1

        action = self.predict()

        if args.forcast: print(f'{self.datas[0].datetime.date(0)} Final Action: {action}')
        
        # sell if current drawdown is more than 20%
        if args.drawdown and self.broker.getvalue() < self.sellcriteria*0.8 and stake > 0:
            self.sellcriteria = self.broker.getvalue()
            action = 'sell'
            sellamount = stake
            if self.datas[0].datetime.date(0) == self.final_date and args.forcast:
                print(f'{self.datas[0].datetime.date(0)} Sell due to drawdown: {self.broker.getvalue()}')


        # Check if we are in the market
        # if not self.position:
        if action == 'buy' and buyamount > 0:
            self.log('BUY CREATE, %.2f' % self.dataclose[0])
            # Keep track of the created order to avoid a 2nd order
            self.order = self.buy(size=buyamount)
        # else:
        elif action == 'sell' and sellamount > 0:
            self.log('SELL CREATE, %.2f' % self.dataclose[0])
            # Keep track of the created order to avoid a 2nd order
            self.order = self.sell(size=sellamount)

        # record highest and lowest portfolio value
        self.highest = max(self.highest, self.broker.getvalue())
        self.lowest = min(self.lowest, self.broker.getvalue())
        self.sellcriteria = max(self.sellcriteria, self.broker.getvalue())

    def stop(self):
        
        # try:
        # # self.analyzers.returns.stop()
        # self.analyzers.vwr.stop()
        # sharpe_ratio = self.analyzers.vwr.get_analysis()['vwr']
        # self.analyzers.mycalmar.stop()
        # except:
        #     print('error of vwr')
        self.analyzers.sharperatio_a.stop()
        sharpe_ratio = self.analyzers.sharperatio_a.get_analysis()['sharperatio']
        calmar = self.analyzers.mycalmar.calmar
        maxdrawndown = self.analyzers.mydrawdown.get_analysis()['max']['drawdown']
        if sharpe_ratio is None: sharpe_ratio = 0
        self.log('(beta %.2f) Ending Value %.2f Highest %.2f Lowest %.2f sharperatio %.2f max drawndown %.2f' %
                 (self.params.beta, self.broker.getvalue(),self.highest,self.lowest,sharpe_ratio,maxdrawndown), doprint=True)
        if args.forcast: return
        # append results to csv file
        with open(os.path.join(args.save_path,f'{args.method}_results.csv'), 'a') as f:
            f.write(f'{self.company},{self.params.whichyhat},{self.params.beta},{self.broker.getvalue()},{self.highest},{self.lowest},{sharpe_ratio},{maxdrawndown}\n')
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
    parser.add_argument('--beta',type=float, default=0.2 ,help='higher than beta %, buy')
    parser.add_argument('--folder_mode',action='store_true' ,default=False,help='run data set in folder')
    parser.add_argument('--plot',action='store_true' ,default=False,help='plot')
    parser.add_argument('--config',default='/zhome/dc/1/174181/docs/QT/code/setting.yml',help='maperiod config')
    parser.add_argument('--save_path',default='/zhome/dc/1/174181/docs/QT/results',help='save path')
    parser.add_argument('--forcast',action='store_true' ,default=False,help='forcast today')
    parser.add_argument('--drawdown',action='store_true' ,default=False,help='sell if drawdown')

    return parser.parse_args()


def get_ts(df):
    '''
    get df for training prophet
    '''
    ts = df.reset_index()[['Date', 'Close']]
    ts.columns = ['ds', 'y']
    ts['ds'] = pd.to_datetime(ts['ds'])
    return ts

def get_realtime_data(c,df):
    '''
    c: company name
    df: original dataframe
    '''
    try:
        today = datetime.date.today()
        price = yf.Ticker(c).info['bid']
        add_dict = {'Date':pd.to_datetime(today),'Close':price}
        tmp = pd.DataFrame(add_dict,index=[0])
        tmp.set_index('Date',inplace=True)
        df = pd.concat([df,tmp],axis=0)
        df.fillna(method='ffill',inplace=True)
    except:
        print(f'{c} No realtime data')
    return df

def main(args):
    datapath = args.data
    filename = os.path.basename(datapath).split('.')[0]
    # Create a cerebro entity
    cerebro = bt.Cerebro()
    df = pd.read_csv(args.data,index_col=0,parse_dates=True)
    if args.forcast:
        df = get_realtime_data(filename,df)

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
            strats = cerebro.optstrategy(myStrategy, beta=[0.05,0.1,0.15,0.2,0.25,0.3])
        elif args.method =='vote':
            strats = cerebro.optstrategy(myStrategy, maperiod=range(10, 31,2), beta=args.beta)
    else:
        cerebro.addstrategy(myStrategy,beta=args.beta)
    # 
    # modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    # datapath = os.path.join(modpath, '../../datas/orcl-1995-2014.txt')
    
    # Create a Data Feed
    # data = bt.feeds.YahooFinanceCSVData(
    #     dataname=datapath,
    #     # Do not pass values before this date
    #     fromdate=fromdate,
    #     # Do not pass values before this date
    #     todate=todate,
    #     # Do not pass values after this date
    #     reverse=False)

    data = bt.feeds.PandasData(dataname=df,fromdate=fromdate,todate=todate)
    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)

    # Add a FixedSize sizer according to the stake
    # cerebro.addsizer(bt.sizers.FixedSize, stake=10)

    # Set the commission
    cerebro.broker.setcommission(commission=0.001)

    #############vwr
    # cerebro.addanalyzer(bt.analyzers.Returns)  # Returns
    # cerebro.addanalyzer(bt.analyzers.SQN)  # VWR Analyzer
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A)  # VWR Analyzer
    cerebro.addanalyzer(bt.analyzers.VWR)  # VWR Analyzer
    # cerebro.addanalyzer(bt.analyzers.TimeReturn,
    #                     timeframe=bt.TimeFrame.Months)
    # cerebro.addanalyzer(bt.analyzers.TimeReturn,
    #                     timeframe=bt.TimeFrame.Years)
    ###############
    cerebro.addanalyzer(bt.analyzers.Calmar,_name='mycalmar')
    # add drawdown
    cerebro.addanalyzer(bt.analyzers.DrawDown,_name='mydrawdown')

    # Run over everything
    cerebro.run()
    # print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    if not args.optimize and args.plot:
        cerebro.plot()
        plt.savefig(os.path.join(args.save_path,f'{filename}_{args.method}.png'))


if __name__ == '__main__':
    args = parse_args()
    # create new csv file
    if not args.forcast:
        with open(os.path.join(args.save_path,f'{args.method}_results.csv'), 'w') as f:
            f.write('company,maperiod,beta,ending value,highest,lowest,sharp_ratio,max_drawndown,calmar\n')
        f.close()

    file = args.config
    with open(file) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    file.close()

    if args.folder_mode:
        folder = args.data
        for file in (os.listdir(folder)):
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