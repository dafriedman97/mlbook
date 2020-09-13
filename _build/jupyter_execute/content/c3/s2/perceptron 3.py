# The Perceptron Algorithm

import numpy as np 
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

# import data
cancer = datasets.load_breast_cancer()
X = cancer['data']
y = cancer['target']

Before constructing the perceptron, let's define a few helper functions. The `sign` function returns `1` for positive numbers and `-1` for non-positive numbers, which will be useful since the perceptron classifies according to 

$$
\text{sign}(\bbeta^\top \bx_n).
$$

Next, the `to_binary` function can be used to convert predictions in $\{-1, +1\}$ to their equivalents in $\{0, 1\}$, which is useful since the perceptron algorithm uses the former though binary data is typically stored as the latter. Finally, the `standard_scaler` standardizes our features, similar to `scikit-learn`'s `StandardScaler`. 


```{note}
Note that we don't actually need to use the `sign` function. Instead, we could deem an observation correctly classified if $y_n \hat{y}_n \geq 0$ and misclassified otherwise. We use it here to be consistent with the derivation in the content section.
```

def sign(a):
    return (-1)**(a < 0)

def to_binary(y):
        return y > 0 

def standard_scaler(X):
    mean = X.mean(0)
    sd = X.std(0)
    return (X - mean)/sd

The perceptron is implemented below. As usual, we optionally standardize and add an intercept term. Then we fit $\bbetahat$ with the algorithm introduced in the {doc}`concept section </content/c3/s2/perceptron>`. 

This implementation tracks whether the perceptron has converged (i.e. all training algorithms are fitted correctly) and stops fitting if so. If not, it will run until `n_iters` is reached. 

class Perceptron:

    def fit(self, X, y, n_iter = 10**3, lr = 0.001, add_intercept = True, standardize = True):
        
        # Add Info #
        if standardize:
            X = standard_scaler(X)
        if add_intercept:
            ones = np.ones(len(X)).reshape(-1, 1)
        self.X = X
        self.N, self.D = self.X.shape
        self.y = y
        self.n_iter = n_iter
        self.lr = lr
        self.converged = False
        
        # Fit #
        beta = np.random.randn(self.D)/5
        for i in range(int(self.n_iter)):
            
            # Form predictions
            yhat = to_binary(sign(np.dot(self.X, beta)))
            
            # Check for convergence
            if np.all(yhat == sign(self.y)):
                self.converged = True
                self.iterations_until_convergence = i
                break
                
            # Otherwise, adjust
            for n in range(self.N):
                yhat_n = sign(np.dot(beta, self.X[n]))
                if (self.y[n]*yhat_n == -1):
                    beta += self.lr * self.y[n]*self.X[n]

        # Return Values #
        self.beta = beta
        self.yhat = to_binary(sign(np.dot(self.X, self.beta)))
                    

Now we can fit the model. We'll again use the {doc}`breast cancer </content/appendix/data>` dataset from `sklearn.datasets`. We can also check whether the perceptron converged and, if so, after how many iterations.

perceptron = Perceptron()
perceptron.fit(X, y, n_iter = 1e3, lr = 0.01)


if perceptron.converged:
    print(f"Converged after {perceptron.iterations_until_convergence} iterations")
else:
    print("Not converged")

np.mean(perceptron.yhat == perceptron.y)