# %%
# 加载框架处理好的数据集
# from sklearn.datasets import load_diabetes
# X1, y1 = load_diabetes(return_X_y=True)
# %% 加载自己的数据集  https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
import numpy as np
datasets = np.loadtxt("diabetes.txt")
X = datasets[:, :10]  # 取前10个数据作为输入
y = datasets[:, -1]  # 取最后1个数据作为输出
X_mean = np.mean(X, axis=0)
X_centered = X - X_mean # 求均值差
std = np.std(X_centered, axis=0) # 求标准差
# 对数据依照方差进行缩放
scaled_X = X_centered * std 
scale = np.sqrt(np.sum(scaled_X**2, axis=0))
scaled_X = scaled_X/scale
# %% 随机森林训练和预测 =============================================== 
from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor(random_state=1)
reg_rf.fit(scaled_X, y)
# 新数据预测
x_new = [[49,1,31.1,110,154,95.2,33,4,4.6692,97]]
scaled_x_new = (x_new - X_mean) * std/scale
pred_rf = reg_rf.predict(scaled_x_new)
print(pred_rf)
# %% 投票回归器训练和预测 =============================================== 
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
reg1 = GradientBoostingRegressor(random_state=1)
reg2 = RandomForestRegressor(random_state=1)
reg3 = LinearRegression()
reg1.fit(scaled_X, y)
reg2.fit(scaled_X, y)
reg3.fit(scaled_X, y)

ereg = VotingRegressor([('gb', reg1), ('rf', reg2), ('lr', reg3)])
ereg.fit(scaled_X, y)
# %% 预测
scaled_x_new = scaled_X[:15]
pred1 = reg1.predict(scaled_x_new)
pred2 = reg2.predict(scaled_x_new)
pred3 = reg3.predict(scaled_x_new)
pred4 = ereg.predict(scaled_x_new)
# %% 绘图
plt.figure()
plt.plot(pred1, 'gd', label='GradientBoostingRegressor')
plt.plot(pred2, 'b^', label='RandomForestRegressor')
plt.plot(pred3, 'ys', label='LinearRegression')
plt.plot(pred4, 'r*', ms=10, label='VotingRegressor')
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.ylabel('predicted')
plt.xlabel('training samples')
plt.legend(loc="best")
plt.title('Regressor predictions and their average')
plt.show()