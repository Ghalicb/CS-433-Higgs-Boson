
from implementations import *
import numpy as np

x = np.arange(1,11)
y = 3 * x

# print(len(np.shape(x)))
(w, loss) = ridge_regression(y, x, 3)
(w_ls, loss_ls) = least_squares(y, x)

print("w, loss =", (w, loss))
print("w_ls, loss_ls =", (w_ls, loss_ls))
