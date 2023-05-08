from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from T2Cap import model_CapDec, model_CapC1
import pandas as pd
import numpy as np


df = pd.read_csv('data0504.csv', encoding='gbk', low_memory=False, parse_dates=['time'], index_col=['time'])





# 取滑动窗口均值
def MeanFilter(data, window_size):

    # 使用 rolling 和 mean 函数对数据进行滤波
    filtered_data = data.rolling(window_size, center=True).mean()

    # 返回滤波后的数据
    return filtered_data

# 滑动窗口均值
mean_window_size = 10




def data_init():
    data_MeanFiltered = df.copy()
    for i in data_MeanFiltered.columns:
        data_MeanFiltered[i] = MeanFilter(data_MeanFiltered[i], mean_window_size)

    data_MeanFiltered = data_MeanFiltered.shift(-5)
    # print(data_MeanFiltered)
    # 实验中所用到的所有时间段选择
    ChosenRange_start = pd.Timestamp('2022-01-01 00:00:00')
    ChosenRange_end = pd.Timestamp('2022-03-06 00:05:00')
    # ChosenRange_end = pd.Timestamp('2022-01-04 00:00:00')
    # 这之前的数据用于构建模型池,之后用于测试
    test_start_ts = pd.Timestamp('2022-01-02 00:00:00')

    data = data_MeanFiltered.loc[ChosenRange_start:ChosenRange_end, ]
    # print(data)
    # while True:
    #     pass
    ploy_T = PolynomialFeatures(degree=1)
    T_Dec_test = ploy_T.fit_transform(data['分解炉出口温度'].values.reshape(-1,1))
    T_C1_test = ploy_T.fit_transform(data['C1出口温度'].values.reshape(-1,1))
    Cap_Dec_ = model_CapDec.predict(T_Dec_test)
    Cap_C1_ = model_CapC1.predict(T_C1_test)
    T_Dec_ = np.delete(T_Dec_test, 0, axis=1).reshape(1, -1)[0]
    T_C1_ = np.delete(T_C1_test, 0, axis=1).reshape(1, -1)[0]
    y = (0.95*T_Dec_*Cap_Dec_ - T_C1_*Cap_C1_)/(0.95*T_Dec_*Cap_Dec_)
    data.insert(loc=len(df.columns), column='预热器热效率', value=y)
    # data.insert(loc=len(df.columns), column='C1出口比热容', value=Cap_C1_pred)

    return data
