# Implementation

This section shows how the linear regression extensions discussed in this chapter are typically fit in Python. First let's import the {doc}`Boston housing</content/appendix/data>` dataset.

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
boston = datasets.load_boston()
X_train = boston['data']
y_train = boston['target']

## Regularized Regression

Both Ridge and Lasso regression can be easily fit using `scikit-learn`. A bare-bones implementation is provided below. Note that the regularization parameter `alpha` (which we called $\lambda$) is chosen arbitrarily.

from sklearn.linear_model import Ridge, Lasso
alpha = 1

# Ridge
ridge_model = Ridge(alpha = alpha)
ridge_model.fit(X_train, y_train)


# Lasso
lasso_model = Lasso(alpha = alpha)
lasso_model.fit(X_train, y_train);

In practice, however, we want to choose `alpha` through cross validation. This is easily implemented in `scikit-learn` by designating a set of `alpha` values to try and fitting the model with `RidgeCV` or `LassoCV`. 

from sklearn.linear_model import RidgeCV, LassoCV
alphas = [0.01, 1, 100]

# Ridge
ridgeCV_model = RidgeCV(alphas = alphas)
ridgeCV_model.fit(X_train, y_train)

# Lasso
lassoCV_model = LassoCV(alphas = alphas)
lassoCV_model.fit(X_train, y_train);

We can then see which values of `alpha` performed best with the following.

print('Ridge alpha:', ridgeCV.alpha_)
print('Lasso alpha:', lassoCV.alpha_)

## Bayesian Regression

We can also fit Bayesian regression using `scikit-learn` (though another popular package is `pymc3`). A very straightforward implementation is provided below. 

from sklearn.linear_model import BayesianRidge
bayes_model = BayesianRidge()
bayes_model.fit(X_train, y_train);

This is not, however, identical to our construction in the previous section since it infers the $\sigma^2$ and $\tau$ parameters, rather than taking those as fixed inputs. More information can be found [here](https://scikit-learn.org/stable/modules/linear_model.html#bayesian-regression). The hidden chunk below demonstrates a hacky solution for running Bayesian regression in `scikit-learn` using known values for $\sigma^2$ and $\tau$, though it is hard to imagine a practical reason to do so

````{toggle}
By default, Bayesian regression in `scikit-learn` treats $\alpha = \frac{1}{\sigma^2}$ and $\lambda = \frac{1}{\tau}$ as random variables and assigns them the following prior distributions

$$
\begin{aligned}
\alpha &\sim \text{Gamma}(\alpha_1, \alpha_2) 
\\
\lambda &\sim \text{Gamma}(\lambda_1, \lambda_2).
\end{aligned}
$$

Note that $E(\alpha) = \frac{\alpha_1}{\alpha_2}$ and $E(\lambda) = \frac{\lambda_1}{\lambda_2}$. To *fix* $\sigma^2$ and $\tau$, we can provide an extremely strong prior on $\alpha$ and $\lambda$, guaranteeing that their estimates will be approximately equal to their expected value.

Suppose we want to use $\sigma^2 = 11.8$ and $\tau = 10$, or equivalently $\alpha = \frac{1}{11.8}$, $\lambda = \frac{1}{10}$. Then let

$$
\begin{aligned}
\alpha_1 &= 10000 \cdot \frac{1}{11.8}, \\
\alpha_2 &= 10000, \\
\lambda_1 &= 10000 \cdot \frac{1}{10}, \\
\lambda_2 &= 10000.
\end{aligned}
$$

This guarantees that $\sigma^2$ and $\tau$ will be approximately equal to their pre-determined values. This can be implemented in `scikit-learn` as follows

```{code}
big_number = 10**5

# alpha
alpha = 1/11.8
alpha_1 = big_number*alpha
alpha_2 = big_number

# lambda 
lam = 1/10
lambda_1 = big_number*lam
lambda_2 = big_number

# fit 
bayes_model = BayesianRidge(alpha_1 = alpha_1, alpha_2 = alpha_2, alpha_init = alpha,
                     lambda_1 = lambda_1, lambda_2 = lambda_2, lambda_init = lam)
bayes_model.fit(X_train, y_train);
```

````

## Poisson Regression

GLMs are most commonly fit in Python through the `GLM` class from `statsmodels`. A simple Poisson regression example is given below.

As we saw in the GLM concept section, a GLM is comprised of a random distribution and a link function. We identify the random distribution through the `family` argument to `GLM` (e.g. below, we specify the `Poisson` family). The default link function depends on the random distribution. By default, the Poisson model uses the link function

$$
\eta_n = g(\mu_n) = \log(\lambda_n),
$$

which is what we use below. For more information on the possible distributions and link functions, check out the `statsmodels` GLM [docs](https://www.statsmodels.org/stable/glm.html).

import statsmodels.api as sm
X_train_with_constant = sm.add_constant(X_train)

poisson_model = sm.GLM(y_train, X_train, family=sm.families.Poisson())
poisson_model.fit();