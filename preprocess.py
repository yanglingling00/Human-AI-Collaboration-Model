# -*- coding:utf-8 -*-
# datetime:2024/8/19 9:56
# software: PyCharm
# introduce: 提取6、12、24  3个特征维度 分别进行训练
import sys

import talib
import pandas as pd
import pywt
import matplotlib.pyplot as plt


def get_indicators(currency, time_frame):
    """
    计算技术指标
    :param currency: 货币对
    :param time_frame: 时间间隔 不同交易周期的数据时间长度不同
    """
    periods = [6, 12, 24]
    # 三个周期进行指标提取
    for period in periods:
        # 选择数据
        if time_frame == '1_D':
            history = pd.read_csv(f'{currency}_Candlestick_{time_frame}_BID_02.01.2013-31.12.2023.csv')
        else:
            history = pd.read_csv(f'{currency}_Candlestick_{time_frame}_BID_02.01.2010-31.12.2023.csv')


        # 计算技术指标
        close_price = history['Close'].values
        high_price = history['High'].values
        low_price = history['Low'].values
        volume = history['Volume'].values

        # PPC  Lagged_PPC
        history['PPC'] = history['Close'].pct_change() * 100
        history['Lagged_PPC'] = history['PPC'].shift()
        # EMA   macd, macdsignal, macdhist   BBANDS RSI
        # AD = talib.AD(high_price,low_price,close_price,volume)
        EMA = talib.EMA(close_price, timeperiod=period)
        macd, macdsignal, macdhist = talib.MACD(close_price, fastperiod=period, slowperiod=int(period * 2),
                                                signalperiod=9)
        upper, middle, lower = talib.BBANDS(close_price, period, matype=talib.MA_Type.EMA)
        Williams = talib.WILLR(history['High'].values, history['Low'].values, close_price, timeperiod=period)
        RSI = talib.RSI(close_price, timeperiod=period)
        # CCI = talib.CCI(high_price,low_price,close_price,timeperiod=period)
        # fast_k,fast_d = talib.STOCHF(high_price,low_price,close_price,fastk_period=period,fastd_period=3)
        # history['AD'] = AD
        history['EMA'] = EMA
        history['macd'] = macdhist
        history['upper'] = upper
        history['middle'] = middle
        history['lower'] = lower
        history['Williams'] = Williams
        history['RSI'] = RSI
        # history['CCI'] = CCI
        # history['fast_k'] = fast_k
        # history['fast_d'] = fast_d

        # 删除'Volume'列
        history = history.drop('Volume', axis=1)
        # 保存结果
        history.to_excel(f'period_{period}_GMT_{currency}_{time_frame}.xlsx', index=False)


def pre_process_data(currency, time_frame):
    """
    生成label   删除空数据
    :param currency:  货币对
    :param time_frame:  时间间隔
    :return:
    """

    # 读取数据
    short_data = pd.read_excel(f"period_6_GMT_{currency}_{time_frame}.xlsx")
    middle_data = pd.read_excel(f"period_12_GMT_{currency}_{time_frame}.xlsx")
    long_data = pd.read_excel(f"period_24_GMT_{currency}_{time_frame}.xlsx")
    # 找到第一个非空value值对应的index
    target_index = long_data['macd'].first_valid_index()
    # 删除到target_index的所有行
    if target_index is not None:
        short_data = short_data.loc[target_index:]
        middle_data = middle_data.loc[target_index:]
        long_data = long_data.loc[target_index:]
    # 生成label
    short_data['lable'] = (short_data['Close'].shift(-1) > short_data['Close']).astype(int)
    middle_data['lable'] = (middle_data['Close'].shift(-1) > middle_data['Close']).astype(int)
    long_data['lable'] = (long_data['Close'].shift(-1) > long_data['Close']).astype(int)
    # 保存
    short_data.to_excel(f"period_6_GMT_{currency}_{time_frame}.xlsx")
    middle_data.to_excel(f"period_12_GMT_{currency}_{time_frame}.xlsx")
    long_data.to_excel(f"period_24_GMT_{currency}_{time_frame}.xlsx")


def wavelet_no_noise(period, currency, time_frame):
    """
    DWT  去噪
    :param period:
    :param currency:
    :param time_frame:
    :return:
    """
    # Get data:
    col = ["Close", "PPC", "Lagged_PPC", "EMA", "macd", "upper", 'middle', 'lower', 'Williams', 'RSI']
    dataSet = pd.read_excel(f"period_{period}_GMT_{currency}_{time_frame}.xlsx")
    for column in col:
        data = dataSet[column].values
        index = [i for i in range(len(data))]
        # 小波类和参数
        w = pywt.Wavelet('db8')  # 选用Daubechies8小波
        t = w.dec_len
        maxlev = pywt.dwt_max_level(len(data), w.dec_len)  # 计算最大有用分解级别。
        threshold = 0.07  # Threshold for filtering 过滤域值
        coeffs = pywt.wavedec(data, 'db8', level=maxlev)  # 将信号进行小波分解

        # 可视化对比
        plt.figure()
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))  # 将噪声滤波
        datarec = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构
        mintime = 0
        maxtime = mintime + len(data) + 1
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(index[mintime:maxtime], data[mintime:maxtime])
        plt.xlabel('time (s)')
        plt.ylabel('microvolts (uV)')
        plt.title("{}Raw signal".format(column))
        plt.subplot(2, 1, 2)
        plt.plot(index[mintime:maxtime], datarec[mintime:maxtime - 1])
        plt.xlabel('time (s)')
        plt.ylabel('microvolts (uV)')
        plt.title("De-noised signal using wavelet techniques")

        plt.tight_layout()
        plt.show()
        a = column
        dataSet[a] = datarec[mintime:maxtime - 1]
    # 保存结果
    dataSet.to_excel(f"period_{period}_GMT_{currency}_{time_frame}.xlsx")


if __name__ == '__main__':
    period_list = [6, 12, 24]
    currency = 'EURUSD'
    time_frame = "15_M"
    # 计算不同维度技术指标
    get_indicators(currency, time_frame)
    # 预处理 合并提取标签
    pre_process_data(currency, time_frame)
    # 去噪
    for period in period_list:
        wavelet_no_noise(period, currency, time_frame)
