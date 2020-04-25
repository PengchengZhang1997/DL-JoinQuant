# 导入函数库
from jqdata import *
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import pandas as pd
from six import BytesIO
import talib as tb

# 初始化函数，设定基准等等
def initialize(context):
    data_dim = 14
    timesteps = 8
    num_classes = 2
    g.security=pd.read_csv(BytesIO(read_file('sec1.csv')))
    dataset1 = pd.read_csv(BytesIO(read_file('train_x1.csv')))
    dataset2 = pd.read_csv(BytesIO(read_file('train_y1.csv')))
    train_x=dataset1.values
    train_y=dataset2.values
    x_train=train_x.flatten()
    y_train=train_y.flatten()
    x_train=x_train.reshape((124200, timesteps, data_dim))
    y_train=y_train.reshape((124200, 2))
    print(y_train.shape)
    # 期望输入数据尺寸: (batch_size, timesteps, data_dim)
    g.model = Sequential()
    g.model.add(LSTM(16, return_sequences=True,input_shape=(timesteps, data_dim)))  # 返回维度为 32 的向量序列
    g.model.add(LSTM(16, return_sequences=True))  # 返回维度为 16 的向量序列
    g.model.add(LSTM(16))  # 返回维度为 16 的单个向量
    g.model.add(Dense(2, activation='softmax'))
    g.model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    g.model.fit(x_train, y_train,batch_size=1242, epochs=10)
    g.cash = context.portfolio.available_cash
    run_daily(period1,time='14:50')
    run_daily(period,time='14:50')

def get_macd(stock):
    array = get_bars(security=stock,count=300,unit='1d',fields=['close'],include_now=True,end_dt=None,fq_ref_date=None)
    close_list = array['close']
    dif, dea, macd = tb.MACD(close_list, fastperiod=12, slowperiod=26, signalperiod=9)
    if(dif[len(dif)-1]>dea[len(dea)-1] and dif[len(dif)-2]<dea[len(dea)-2]):
        #macd金叉
        return 1
    elif(dif[len(dif)-1]<dea[len(dea)-1] and dif[len(dif)-2]>dea[len(dea)-2]):
        #macd死叉
        return -1
    else:
        return 0

def period(context):
    model=g.model
    arr=[]
    for i in range(len(g.security)):
        val=[]
        sec=g.security['sec'][i]
        macd=get_macd(sec)
        if(sec in context.portfolio.positions.keys()):
            continue
        mtss1=get_mtss(sec, start_date='2015-01-01', end_date=context.current_dt)
        mtss1=mtss1.set_index('date')
        mi=get_price('399001.XSHE',start_date='2015-01-01',end_date=context.current_dt,frequency='daily',fields=['close'],skip_paused=False,fq='pre',panel=True)
        del mtss1['sec_code']
        mtss1.insert(7,'market_index',mi['close'])
        info=get_price(sec,start_date='2015-01-01',end_date=context.current_dt,frequency='daily',skip_paused=False,fq='pre',panel=True)
        mtss1.insert(8,'open',info['open'])
        mtss1.insert(9,'close',info['close'])
        mtss1.insert(10,'high',info['high'])
        mtss1.insert(11,'low',info['low'])
        mtss1.insert(12,'volume',info['volume'])
        mtss1.insert(13,'money',info['money'])
        mtss1 = (mtss1 - mtss1.min()) / (mtss1.max() - mtss1.min())
        test_x=mtss1.ix[-8:]
        test_x=test_x.values
        x_test=[]
        x_test.append(test_x)
        x_test=np.array(x_test)
        updown=model.predict(x_test)
        cash = g.cash/len(g.security)
        volume=get_price(sec,start_date='2015-01-01',end_date=context.current_dt,frequency='daily',skip_paused=False,fq='pre',panel=True)
        if(updown[0][0]<updown[0][1] and macd==1):#预测买入且金叉则买，也可或时买入
            if(sec not in context.portfolio.positions.keys()):
                vol=-volume['volume'][len(volume)-1]
                val.append(vol)
                val.append(sec)
                arr.append(val)
    #arr.sort()
    axx=len(arr)
    for i in range(axx):
        sec=arr[i][1]
        order_value(sec, cash)
        log.info("Buying %s" % (sec))

def period1(context):
    model=g.model
    for i in range(len(g.security)):
        sec=g.security['sec'][i]
        macd=get_macd(sec)
        if(sec not in context.portfolio.positions.keys()):
            continue
        mtss1=get_mtss(sec, start_date='2015-01-01', end_date=context.current_dt)
        mtss1=mtss1.set_index('date')
        mi=get_price('399001.XSHE',start_date='2015-01-01',end_date=context.current_dt,frequency='daily',fields=['close'],skip_paused=False,fq='pre',panel=True)
        del mtss1['sec_code']
        mtss1.insert(7,'market_index',mi['close'])
        info=get_price(sec,start_date='2015-01-01',end_date=context.current_dt,frequency='daily',skip_paused=False,fq='pre',panel=True)
        mtss1.insert(8,'open',info['open'])
        mtss1.insert(9,'close',info['close'])
        mtss1.insert(10,'high',info['high'])
        mtss1.insert(11,'low',info['low'])
        mtss1.insert(12,'volume',info['volume'])
        mtss1.insert(13,'money',info['money'])
        mtss1 = (mtss1 - mtss1.min()) / (mtss1.max() - mtss1.min())
        test_x=mtss1.ix[-8:]
        test_x=test_x.values
        x_test=[]
        x_test.append(test_x)
        x_test=np.array(x_test)
        updown=model.predict(x_test)
        cash = g.cash/len(g.security)
        if(updown[0][0]>updown[0][1] and macd==-1):#预测卖出且死叉则卖出，也可或时卖出
            order_target(sec, 0)
            log.info("Selling %s" % (sec))
