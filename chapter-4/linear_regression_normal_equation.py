import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from save_plt_as_image import save_fig

np.random.seed(42)
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures


X = 4 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# print('X = ', X)
# print('Y = ', y)
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 4, 0, 15])
save_fig("generated_data_plot")
plt.show()

X_b = np.c_[np.ones((100, 1)), X]
# print('X_b = ', X_b)
# equation
#  theta = (X.T * X)^-1 * (X.T *y)
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

X_new = np.array([[0], [4]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)
# print('y_predict', y_predict)
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 4, 0, 15])
save_fig("predict_out_put")
plt.show()


###############################################################################
print('------------USING Scikit----------------')
###############################################################################

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print('scikit coef(theta)=',lin_reg.intercept_, lin_reg.coef_)
print('theta_best =',theta_best)

lin_reg.predict(X_new)

theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
print('theta_best_svd', theta_best_svd)