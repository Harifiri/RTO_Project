from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import pandas as pd


# 初始模型训练集长度
train_size_num = 22

# MSE检测长度
MSE_check_size = 10

# MSE检测阈值
mse_check_threshold = 0.005*0.005

# 模型次数
degree_num = 1
ploy = PolynomialFeatures(degree=degree_num)

# 在线建模训练集的搜索参数
train_set_size_lb = 10
train_set_size_ub = 100
train_set_size_gap = 2
# 特征选择
inputs = ['分解炉出口温度', 'C1出口温度', 'C1出口O2', '二次风温', 'F1234平均流量', 'F56789平均流量']
outputs = ['预热器热效率']

# 返回模型输入数据
def Xdata(data, start_ts, end_ts):
    data_trainingset_list = data.loc[start_ts:end_ts, :]
    data_trainingset_list = data_trainingset_list.fillna(method='ffill').fillna(method='bfill')
    X = data_trainingset_list[inputs].values
    return X


#  返回模型输出数据
def ydata(data, start_ts, end_ts):
    data_trainingset_list = data.loc[start_ts:end_ts, :]
    data_trainingset_list = data_trainingset_list.fillna(method='ffill').fillna(method='bfill')
    y = data_trainingset_list[outputs].values
    y = y.flatten()
    return y


#  返回岭回归模型
def RidgeRegressionModel(data, start_ts, end_ts, poly_degree=2, set_alpha=0.02):
    data_trainingset_list = data.loc[start_ts:end_ts, :]
    # print(data_trainingset_list)
    data_trainingset_list = data_trainingset_list.fillna(method='ffill').fillna(method='bfill')
    # print(data_trainingset_list)
    X = data_trainingset_list.loc[:, inputs]
    # print(X)
    y = data_trainingset_list[outputs].values
    y = y.flatten()

    ploy = PolynomialFeatures(poly_degree)
    X_new = ploy.fit_transform(X)
    model = Ridge(alpha=set_alpha)
    model.fit(X_new, y)
    return model


#  返回模型在某段时间上的MSE
def RidgeModelCheckMSE(model, data, current_time, back_check_window_size):
    X_check_model = Xdata(data, current_time - pd.Timedelta(minutes=back_check_window_size), current_time)
    y_check_model = ydata(data, current_time - pd.Timedelta(minutes=back_check_window_size), current_time)
    y_check_model_pred = model.predict(ploy.fit_transform(X_check_model))
    mse_check_model = float(mean_squared_error(y_check_model_pred, y_check_model))
    return mse_check_model


#  挑选模型池中最好的模型
def choose_model(model_pool, data, time_now, MSE_check_size):
    for index, row in model_pool.iterrows():
        model = row['model']
        mse_check_model = RidgeModelCheckMSE(model, data, time_now, MSE_check_size)
        model_pool.loc[index, 'MSE'] = mse_check_model
    model_pool['MSE'] = model_pool['MSE'].astype(float)
    best_model_index = model_pool['MSE'].idxmin()
    return model_pool.loc[best_model_index, 'model'], time_now - pd.Timedelta(minutes=MSE_check_size), time_now

