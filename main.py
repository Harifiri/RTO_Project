from sklearn.preprocessing import PolynomialFeatures
from DataPreparation import *
import time
import pandas as pd
import warnings
from pylab import mpl
from model import *
import matplotlib.pyplot as plt
import sys

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False
pd.set_option('display.max_columns', None) # 显示dataframe类型时 不限制columns标签显示
pd.set_option('display.max_rows', 100) # 显示dataframe类型时 不限制columns标签显示
pd.options.display.max_seq_items = None # df.columns 显示不全
warnings.filterwarnings("ignore")



data1 = data_init()
start_time = time.time()



# 在线建立一个模型
test_start_ts = pd.Timestamp('2022-01-01 00:00:00')
ChosenRange_end = pd.Timestamp('2022-03-06 00:00:00')

original_train_start_ts = test_start_ts - pd.Timedelta(minutes=train_size_num)
original_train_end_ts = test_start_ts

original_model = RidgeRegressionModel(data1, original_train_start_ts, original_train_end_ts, degree_num)

# modeltype = 'ridge'
# rto_size = 60
# test_size_num = mean_window_size + rto_size;

# 事先训练模型池

# 模型池用dataframe存放
# 存放了模型、模型训练集范围。MSE列用于存放MSE_check时候的数据。
model_pool = pd.DataFrame(data=[[original_model, original_train_start_ts, original_train_end_ts, None]],
                          columns=['model', 'start_ts', 'end_ts', 'MSE'])

# 存放了模模型训练集范围、模型类别、模型参数
model_pool_original_trainingset = [
    ['2022-01-01 0:00', '2022-01-01 0:29', 'PolynomialRidge', 2],
    ['2022-01-01 0:30', '2022-01-01 0:59', 'PolynomialRidge', 2],
    ['2022-01-01 1:00', '2022-01-01 1:29', 'PolynomialRidge', 2],
    ['2022-01-01 1:30', '2022-01-01 1:59', 'PolynomialRidge', 2],
    ['2022-01-01 2:00', '2022-01-01 2:29', 'PolynomialRidge', 2],
    ['2022-01-01 2:30', '2022-01-01 2:59', 'PolynomialRidge', 2],
    ['2022-01-01 3:00', '2022-01-01 3:29', 'PolynomialRidge', 2],
    ['2022-01-01 3:30', '2022-01-01 3:59', 'PolynomialRidge', 2],
    ['2022-01-01 4:00', '2022-01-01 4:29', 'PolynomialRidge', 2],
    ['2022-01-01 4:30', '2022-01-01 4:59', 'PolynomialRidge', 2],
    ['2022-01-01 5:00', '2022-01-01 5:29', 'PolynomialRidge', 2],
    ['2022-01-01 5:30', '2022-01-01 5:59', 'PolynomialRidge', 2],
    ['2022-01-01 6:00', '2022-01-01 6:29', 'PolynomialRidge', 2],
    ['2022-01-01 6:30', '2022-01-01 6:59', 'PolynomialRidge', 2],
    ['2022-01-01 7:00', '2022-01-01 7:29', 'PolynomialRidge', 2],
    ['2022-01-01 7:30', '2022-01-01 7:59', 'PolynomialRidge', 2],
    ['2022-01-01 8:00', '2022-01-01 8:29', 'PolynomialRidge', 2],
    ['2022-01-01 8:30', '2022-01-01 8:59', 'PolynomialRidge', 2],
    ['2022-01-01 9:00', '2022-01-01 9:29', 'PolynomialRidge', 2],
    ['2022-01-01 9:30', '2022-01-01 9:59', 'PolynomialRidge', 2],
    ['2022-01-01 10:00', '2022-01-01 10:29', 'PolynomialRidge', 2],
    ['2022-01-01 10:30', '2022-01-01 10:59', 'PolynomialRidge', 2],
    ['2022-01-01 11:00', '2022-01-01 11:29', 'PolynomialRidge', 2],
    ['2022-01-01 11:30', '2022-01-01 11:59', 'PolynomialRidge', 2],
    ['2022-01-01 12:00', '2022-01-01 12:29', 'PolynomialRidge', 2],
    ['2022-01-01 12:30', '2022-01-01 12:59', 'PolynomialRidge', 2],
    ['2022-01-01 13:00', '2022-01-01 13:29', 'PolynomialRidge', 2],
    ['2022-01-01 13:30', '2022-01-01 13:59', 'PolynomialRidge', 2],
    ['2022-01-01 14:00', '2022-01-01 14:29', 'PolynomialRidge', 2],
    ['2022-01-01 14:30', '2022-01-01 14:59', 'PolynomialRidge', 2],
    ['2022-01-01 15:00', '2022-01-01 15:29', 'PolynomialRidge', 2],
    ['2022-01-01 15:30', '2022-01-01 15:59', 'PolynomialRidge', 2],
    ['2022-01-01 16:00', '2022-01-01 16:29', 'PolynomialRidge', 2],
    ['2022-01-01 16:30', '2022-01-01 16:59', 'PolynomialRidge', 2],
    ['2022-01-01 17:00', '2022-01-01 17:29', 'PolynomialRidge', 2],
    ['2022-01-01 17:30', '2022-01-01 17:59', 'PolynomialRidge', 2],
    ['2022-01-01 18:00', '2022-01-01 18:29', 'PolynomialRidge', 2],
    ['2022-01-01 18:30', '2022-01-01 18:59', 'PolynomialRidge', 2],
    ['2022-01-01 19:00', '2022-01-01 19:29', 'PolynomialRidge', 2],
    ['2022-01-01 19:30', '2022-01-01 19:59', 'PolynomialRidge', 2],
    ['2022-01-01 20:00', '2022-01-01 20:29', 'PolynomialRidge', 2],
    ['2022-01-01 20:30', '2022-01-01 20:59', 'PolynomialRidge', 2],
    ['2022-01-01 21:00', '2022-01-01 21:29', 'PolynomialRidge', 2],
    ['2022-01-01 21:30', '2022-01-01 21:59', 'PolynomialRidge', 2],
    ['2022-01-01 22:00', '2022-01-01 22:29', 'PolynomialRidge', 2],
    ['2022-01-01 22:30', '2022-01-01 22:59', 'PolynomialRidge', 2],
    ['2022-01-01 23:00', '2022-01-01 23:29', 'PolynomialRidge', 2],
    ['2022-01-01 23:10', '2022-01-01 23:59', 'PolynomialRidge', 2],
    ['2022-01-01 23:15', '2022-01-01 23:59', 'PolynomialRidge', 2],
    ['2022-01-01 23:20', '2022-01-01 23:59', 'PolynomialRidge', 2],
    ['2022-01-01 23:25', '2022-01-01 23:59', 'PolynomialRidge', 2],
    ['2022-01-01 23:30', '2022-01-01 23:59', 'PolynomialRidge', 2],
    ['2022-01-01 23:35', '2022-01-01 23:59', 'PolynomialRidge', 2],
    ['2022-01-01 23:40', '2022-01-01 23:59', 'PolynomialRidge', 2],
    ['2022-01-01 23:45', '2022-01-01 23:59', 'PolynomialRidge', 2],
]

for trainingset_list in model_pool_original_trainingset:
    start_ts = pd.Timestamp(trainingset_list[0])
    end_ts = pd.Timestamp(trainingset_list[1])
    if trainingset_list[2] == 'PolynomialRidge':
        model = RidgeRegressionModel(data1, start_ts, end_ts, degree_num)
    model_pool.loc[len(model_pool)] = [model, start_ts, end_ts, None]

# print(model_pool)

# 存放真实值
y_real = ydata(data1, test_start_ts, ChosenRange_end)
# print(y_real)


# 存放预测值
y_pred_list = []

# 存放历史状态
model_pool_len_list = []
mse_check_model_list = []

# 存放模型切换信息
model_change_time_list = []

# 存放待检查模型
models_to_be_checked = pd.DataFrame(data=[[None, None, None, None]], columns=['model', 'start_ts', 'end_ts', 'MSE'])
models_to_be_checked = models_to_be_checked.drop(0, axis=0)

current_model = original_model

##########################################################################

def progress_bar(finish_tasks_number, tasks_number):
    """
    进度条

    :param finish_tasks_number: int, 已完成的任务数
    :param tasks_number: int, 总的任务数
    :return:
    """

    percentage = round(finish_tasks_number / tasks_number * 100)
    print("\r进度: {}%: ".format(percentage), "▓" * (percentage // 2), end="")
    sys.stdout.flush()


# 生成pd.timestamp数据的迭代对象
test_range_iteration = pd.date_range(start=test_start_ts, end=ChosenRange_end, freq='1min')

time_lenth = len(test_range_iteration)
time_cnt = 0

for time_now in test_range_iteration:
    progress_bar(time_cnt, time_lenth)
    time_cnt += 1
    model_pool_len_list.append(len(model_pool))

    if len(model_pool) > 25:
        model_pool = model_pool[1:]

    # 用当前模型预测当前数值
    X = Xdata(data1, time_now, time_now)
    y_pred = current_model.predict(ploy.fit_transform(X))
    y_pred_list.append(y_pred)

    mse_check_model = RidgeModelCheckMSE(current_model, data1, time_now, MSE_check_size)
    mse_check_model_list.append(mse_check_model)

    if not (models_to_be_checked.empty):

        bool_a = (time_now - models_to_be_checked['end_ts'] == pd.Timedelta(minutes=mean_window_size + MSE_check_size))
        models_checking = models_to_be_checked[bool_a]
        #         print(models_to_be_checked)
        #         print(bool_a)
        #         print(models_to_be_checked['end_ts']-time_now,'====',pd.Timedelta(minutes = mean_window_size + MSE_check_size))
        if not (models_checking.empty):
            a_new_model, aa, bb = choose_model(models_checking, data1, time_now, MSE_check_size)
            model_pool.loc[len(model_pool)] = [current_model, aa, bb, None]
            models_to_be_checked = models_to_be_checked.drop(models_to_be_checked[bool_a].index)

    if mse_check_model > mse_check_threshold:

        model_change_time_list.append([time_now, 1])

        # 选出模型池中最好的模型
        current_model, aa, bb = choose_model(model_pool, data1, time_now, MSE_check_size)
        del aa, bb

        # 在线构建新的模型

        for new_model_train_size in range(train_set_size_lb, train_set_size_ub, train_set_size_gap):
            new_model_start_ts = time_now - pd.Timedelta(minutes=new_model_train_size)
            new_model_end_ts = time_now
            new_model = RidgeRegressionModel(data1, new_model_start_ts, new_model_end_ts, poly_degree=degree_num)
            models_to_be_checked.loc[len(models_to_be_checked)] = [new_model, new_model_start_ts, new_model_end_ts,
                                                                   None]

    else:
        model_change_time_list.append([time_now, 0])

save_info = {'真实值': y_real, '预测值': y_pred_list, '模型切换信息': model_change_time_list}
save_index = test_range_iteration

save_df = pd.DataFrame(save_info, index=save_index)
save_df.to_csv('generate_data.csv')


data_result = pd.DataFrame(
    np.concatenate((np.array(y_real).reshape(-1, 1), np.array(y_pred_list).reshape(-1, 1)), axis=1),
    columns=['真实值', '预测值'])
new_index = pd.date_range(test_start_ts, ChosenRange_end, freq='T')
data_result.set_index(new_index, inplace=True)
data_result.index.name = 'time'

data_model_change_result = pd.DataFrame(model_change_time_list, columns=['time', '模型切换信息'])
data_model_change_result.set_index('time', inplace=True)

result = pd.merge(data_result, data_model_change_result, left_index=True, right_index=True, how='outer')
result['模型切换信息'].fillna(0, inplace=True)

time_str = str(int(time.time()))
file_name = 'result_' + time_str + '.csv'
# result.to_csv(file_name,encoding='gbk')


fig2, axes2 = plt.subplots(5, 1, figsize=(24, 25))
# 预测结果图
axes2[0].plot(y_real, marker='o', markersize=4)
axes2[0].plot(y_pred_list, marker='o', markersize=2)
# axes2[0].set_ylim(0, 1)

axes2[0].grid()

# 模型切换信息图
axes2[1].plot(result.loc[:, '模型切换信息'], marker='o', markersize=4, c='r')
axes2[1].grid()

# 单个模型延续时间图直方图
model_change_time_list3 = data_model_change_result[data_model_change_result['模型切换信息'] == 1].index


model_change_time_list2 = []
if(len(model_change_time_list3) == 0):
    model_change_time_list2.append(0)
else:
    model_change_time_list2.append((model_change_time_list3[0] - test_start_ts).total_seconds() / 60.0)

for i in range(len(model_change_time_list3)):
    if i == 0:
        continue
    model_change_time_list2.append((model_change_time_list3[i] - model_change_time_list3[i - 1]).total_seconds() / 60.0)
model_change_time_list2.append((ChosenRange_end - model_change_time_list[-1][0]).total_seconds() / 60.0)

max_bins = int(36000 / 60.0)
min_bins = int(0 / 60.0)
bins_gape = int(600 / 60.0)
bins = range(min_bins, max_bins, bins_gape)
axes2[2].hist(model_change_time_list2, bins=bins, alpha=0.5, edgecolor='black')
axes2[2].set_xticks(np.arange(min_bins, max_bins, bins_gape * 2))

# 模型池大小图
axes2[3].plot(model_pool_len_list, marker='o', markersize=2)
axes2[3].grid()

# 单个模型延续时间图频率分布图

length = 600
data_len = len(model_change_time_list2)
count_list = []
for i in range(length):
    count = 1 - sum(1 for x in model_change_time_list2 if x < i) / data_len
    count_list.append(count)
axes2[4].plot(count_list, marker='o', markersize=2)
axes2[4].grid()
axes2[4].set_ylim(-0.05, 1.05)
axes2[4].set_xticks(np.arange(0, length + 1, 100))
for i in range(30, length, 30):
    axes2[4].text(i - 10, count_list[i - 1] - 0.05, '({},{}%)'.format(i, int(count_list[i - 1] * 100)))

end_time = time.time()
run_time = end_time - start_time
print('======================================================')
print('当前输入特征', inputs)
print('当前MSE阈值：{} '.format(mse_check_threshold))
print('程序运行了：{} 秒'.format(run_time))

print('总切换次数：{0} '.format(len(model_change_time_list3)))

average_model_change_time = (ChosenRange_end - test_start_ts).total_seconds() / (len(model_change_time_list3) + 1)
print('平均模型维持时间：{0} 秒，即 {1} 小时'.format(average_model_change_time, average_model_change_time / 3600.0))

mse_result = mean_squared_error(result['真实值'], result['预测值'])
print('MSE: {0}'.format(mse_result))



plt.show()

