
import numpy as np

# Sin function graph
# x = np.linspace(0.0, 2*np.pi, 100, True)
x = np.linspace(0.0, 2*np.pi, 100, True)
print 'type(x): ', type(x)
y_true = np.sin(x) + np.cos(2*x)

# noisy signal
# try Nsample = 10, 15, 30, 100
n_sample = 100
# x_sample = np.linspace(0.0, 2*np.pi, n_sample, True) # correspondent to x
x_sample = np.linspace(0.0, 2*np.pi, n_sample, True) # correspondent to x
# noise = np.random.normal(0, 0.15, len(x_sample)) # standard deviation 0.15
# noise = np.zeros(shape=(1,len(x_sample)))
noise = np.random.normal(0, 0.2, len(x_sample)) # standard deviation 0.15

# y_sample = np.sin(x_sample) + noise # blue point
y_sample = np.sin(x_sample) + np.cos(2*x_sample) + noise

# Try order = 0, 1, 2, 3, 9, 13
import regression as reg
order = 13 # orde approximation
w_stand = reg.standReg(x_sample, y_sample, order) # solve regression using LR Normal
w_dummy = reg.dummyReg(x_sample, y_sample, order)
w_numpy = reg.numpyReg(x_sample, y_sample, order)
w_ridge = reg.ridgeReg(x_sample, y_sample, order)
w_tikho = reg.tikhonovReg(x_sample, y_sample, order)
w_svd = reg.numpySVDReg(x_sample, y_sample, order)


# create model for drawing
y_model_stand = reg.createModel(x, w_stand)
y_model_dummy = reg.createModel(x, w_dummy)
y_model_numpy = reg.createModel(x, w_numpy)
y_model_ridge = reg.createModel(x, w_ridge)
y_model_tikho = reg.createModel(x, w_tikho)
y_model_svd = reg.createModel(x, w_svd)

# X_sample to compute RMSE
y_model_stand_2 = reg.createModel(x_sample, w_stand)
y_model_dummy_2 = reg.createModel(x_sample, w_dummy)
y_model_numpy_2 = reg.createModel(x_sample, w_numpy)
y_model_ridge_2 = reg.createModel(x_sample, w_ridge)
y_model_tikho_2 = reg.createModel(x_sample, w_tikho)
y_model_svd_2 = reg.createModel(x_sample, w_svd)
# y_model_lwlr = reg.createLWLRModel(x, x_sample)

# plot
import matplotlib.pyplot as plt

import plot_routine as plotr

# plotr.plotLR(1, x, y_true, x_sample, y_sample, y_model_stand, order, 'standard', 2)
# plotr.plotLR(2, x, y_true, x_sample, y_sample, y_model_dummy, order, 'dummy', 2)
# plotr.plotLR(3, x, y_true, x_sample, y_sample, y_model_numpy, order, 'numpy', 2)
# plotr.plotLR(4, x, y_true, x_sample, y_sample, y_model_ridge, order, 'ridge', 2)
# plotr.plotLR(5, x, y_true, x_sample, y_sample, y_model_tikho, order, 'tikho', 2)

# PLOT 2 --- figure 11, 22, and 33 are identical
plotr.plotLR(x, y_true, x_sample, y_sample, y_model_stand, y_model_stand_2, order, 'standard', 2)
plotr.plotLR(x, y_true, x_sample, y_sample, y_model_dummy, y_model_dummy_2, order, 'dummy', 2)
plotr.plotLR(x, y_true, x_sample, y_sample, y_model_numpy, y_model_numpy_2, order, 'numpy', 2)
plotr.plotLR(x, y_true, x_sample, y_sample, y_model_ridge, y_model_ridge_2, order, 'ridge', 2)
plotr.plotLR(x, y_true, x_sample, y_sample, y_model_tikho, y_model_tikho_2, order, 'tikho', 2)
plotr.plotLR(x, y_true, x_sample, y_sample, y_model_svd, y_model_svd_2, order, 'svd', 2)
# plotr.plotLR(x, y_true, x_sample, y_sample, y_model_lwlr, order, 'lwlr', 2)

plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111) # default
# ax.plot(x, y_true, 'g-', linewidth=2)
# ax.scatter(x_sample, y_sample, s=50, facecolors='none', edgecolors='b', linewidths=2)
# # ax.plot(x, y_model_stand, 'r--', linewidth=2, label='standard')
# ax.plot(x, y_model_dummy, 'r-', linewidth=1, label='dummy')
# # ax.plot(x, y_model_numpy, 'r-', linewidth=1, label='numpy)
# # ax.plot(x, y_model_ridge, 'r-', linewidth=2, label='ridge')
# # ax.plot(x, y_model_tikho, 'r-', linewidth=2, label='tikho')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.title('Regression with M = ' + str(order) + ', N = ' + str(len(x_sample)))
# plt.legend()
# plt.grid()
# plt.show()
