import matplotlib.pyplot as plt
import numpy as np

def plotLR(x, y_true, x_sample, y_sample, y_model, y_model2, order, labelv, linewidthv):
    figname = '[n' + str(len(x_sample)) + '][m' + str(order) + ']' + labelv
    fig = plt.figure(figname)
    ax = fig.add_subplot(111) # default
    ax.plot(x, y_true, 'g-', linewidth=2)
    ax.scatter(x_sample, y_sample, s=50, facecolors='none', edgecolors='b', linewidths=2)
    ax.plot(x, y_model, 'r-', linewidth=linewidthv, label=labelv)
    plt.xlabel('X')
    plt.ylabel('y')
    rmse = findRmse(y_sample, y_model2) # compute RMSE
    plt.title('Regression with M = ' + str(order) + ', N = ' + str(len(x_sample)) + ' RMSE = ' + str(rmse))
    plt.legend()
    plt.grid()
    return

# find rmse of model based on sample
def findRmse(y_sample, y_model):
    print 'Y_SAMPLE = ' + str(len(y_sample)) + ' Y_MODEL = ' +  str(len(y_model))
    y_err = np.mat(y_model) - np.mat(y_sample)
    y_err = y_err * y_err.T
    err = np.sqrt(y_err)
    return err[0,0]