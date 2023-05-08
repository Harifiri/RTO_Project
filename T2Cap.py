import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression as LR

# 分解炉出口温度
T_Dec = np.array([875,876,877,878,879,880,881,882,883,884,885,886,887,888,889,890,891,892,893,894,895,896,897,898,899,900]).reshape(-1,1)
# 分解炉出口温度对应比热容
Cap_Dec = np.array([1.6734,1.6736,1.6739,1.6741,1.6744,1.6746,1.6748,1.6751,1.6753,
                    1.6756,1.6758,1.676,1.6763,1.6765,1.6768,1.677,1.6772,1.6775,
                    1.6777,1.678,1.6782,1.6784,1.6787,1.6789,1.6792,1.6794])
# C1出口温度
T_C1 = np.array([250,255,260,265,270,275,280,285,290,295,300,305]).reshape(-1,1)
# 对应比热容
Cap_C1 = np.array([1.484,1.4855,1.4869,1.4884,1.4899,1.4914,1.4928,1.4943,1.4958,1.4973,1.4988,1.5001])

ploy_T = PolynomialFeatures(degree=1)

T_Dec_New = ploy_T.fit_transform(T_Dec)
T_C1_New = ploy_T.fit_transform(T_C1)

model_CapDec = LR().fit(T_Dec_New, Cap_Dec)
model_CapC1 = LR().fit(T_C1_New, Cap_C1)