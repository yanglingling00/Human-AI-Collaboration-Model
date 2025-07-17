# -*- coding:utf-8 -*-
# introduce: Extract the three feature dimensions of 6, 12, and 24 for training respectively
import sys
import talib
import pandas as pd
import pywt
import matplotlib.pyplot as plt

def get_indicators(currency, time_frame):
    """
    Calculation technical indicators
    """
    periods = [6, 12, 24]
    for period in periods:
        if time_frame == '1_D':
            history = pd.read_csv(f'{currency}_Candlestick_{time_frame}_BID_02.01.2013-31.12.2023.csv')
        else:
            history = pd.read_csv(f'{currency}_Candlestick_{time_frame}_BID_02.01.2010-31.12.2023.csv')

        # Calculation technical indicators
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
        
        history = history.drop('Volume', axis=1)
        # Save the result
        history.to_excel(f'period_{period}_GMT_{currency}_{time_frame}.xlsx', index=False)

def pre_process_data(currency, time_frame):
    """
    Generate labels and delete empty data
    """
    short_data = pd.read_excel(f"period_6_GMT_{currency}_{time_frame}.xlsx")
    middle_data = pd.read_excel(f"period_12_GMT_{currency}_{time_frame}.xlsx")
    long_data = pd.read_excel(f"period_24_GMT_{currency}_{time_frame}.xlsx")
    
    # the index corresponding to the first non-empty value
    target_index = long_data['macd'].first_valid_index()
    # Delete all rows to target_index
    if target_index is not None:
        short_data = short_data.loc[target_index:]
        middle_data = middle_data.loc[target_index:]
        long_data = long_data.loc[target_index:]
        
    # Generate label
    short_data['lable'] = (short_data['Close'].shift(-1) > short_data['Close']).astype(int)
    middle_data['lable'] = (middle_data['Close'].shift(-1) > middle_data['Close']).astype(int)
    long_data['lable'] = (long_data['Close'].shift(-1) > long_data['Close']).astype(int)
    
    # save result
    short_data.to_excel(f"period_6_GMT_{currency}_{time_frame}.xlsx")
    middle_data.to_excel(f"period_12_GMT_{currency}_{time_frame}.xlsx")
    long_data.to_excel(f"period_24_GMT_{currency}_{time_frame}.xlsx")


def wavelet_no_noise(period, currency, time_frame):
    """
    DWT denoising
    """
    # Get data:
    col = ["Close", "PPC", "Lagged_PPC", "EMA", "macd", "upper", 'middle', 'lower', 'Williams', 'RSI']
    dataSet = pd.read_excel(f"period_{period}_GMT_{currency}_{time_frame}.xlsx")
    for column in col:
        data = dataSet[column].values
        index = [i for i in range(len(data))]
        # Wavelet classes and parameters
        w = pywt.Wavelet('db8')  # Select the Daubechies8 wavelet
        t = w.dec_len
        maxlev = pywt.dwt_max_level(len(data), w.dec_len)  # Calculate the maximum useful decomposition level
        threshold = 0.07  # Threshold for filtering 
        coeffs = pywt.wavedec(data, 'db8', level=maxlev)  # Perform wavelet decomposition on the signal

        # Visual comparison
        plt.figure()
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))  # Filter out the noise
        datarec = pywt.waverec(coeffs, 'db8')  # Perform wavelet reconstruction on the signal
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
    # Save the result
    dataSet.to_excel(f"period_{period}_GMT_{currency}_{time_frame}.xlsx")

if __name__ == '__main__':
    period_list = [6, 12, 24]
    currency = 'EURUSD'
    time_frame = "15_M"
    # Calculate technical indicators of different dimensions
    get_indicators(currency, time_frame)
    # Preprocess and merge to extract labels
    pre_process_data(currency, time_frame)
    # Denoising
    for period in period_list:
        wavelet_no_noise(period, currency, time_frame)
