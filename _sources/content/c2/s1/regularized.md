Regularized Regression
==============

$$
\newcommand{\sumN}{\sum_{n = 1}^N}
\newcommand{\sumn}{\sum_n}
\newcommand{\prodN}{\prod_{n = 1}^N}
\newcommand{\by}{\mathbf{y}} \newcommand{\bX}{\mathbf{X}}
\newcommand{\bx}{\mathbf{x}}
\newcommand{\bbeta}{\boldsymbol{\beta}}
\newcommand{\btheta}{\boldsymbol{\theta}}
\newcommand{\bbetahat}{\boldsymbol{\hat{\beta}}}
\newcommand{\bthetahat}{\boldsymbol{\hat{\theta}}}
\newcommand{\bSigma}{\boldsymbol{\Sigma}}
\newcommand{\bphi}{\boldsymbol{\phi}}
\newcommand{\bPhi}{\boldsymbol{\Phi}}
\newcommand{\bT}{\mathbf{T}}
\newcommand{\dadb}[2]{\frac{\partial #1}{\partial #2}}
\newcommand{\iid}{\overset{\small{\text{i.i.d.}}}{\sim}}
$$

Regression models, especially those fit to high-dimensional data, may be prone to overfitting. One way to ameliorate this issue is by penalizing the magnitude of the $\bbetahat$ coefficient estimates. This has the effect of shrinking these estimates toward 0, which ideally prevents the model from capturing spurious relationships between weak predictors and the target variable. 



This section reviews the two most common methods for regularized regression: Ridge and Lasso.



## Ridge Regression

Like ordinary linear regression, Ridge regression estimates the coefficients by minimizing a loss function on the training data. Unlike ordinary linear regression, the loss function for Ridge regression penalizes large values of the $\bbetahat$ estimates. Specifically, Ridge regression minimizes the sum of the RSS and the L2 norm of $\bbetahat$:


$$
\begin{align*}
\mathcal{L}_\text{Ridge}(\bbetahat) &= \frac{1}{2}\left(\by - \bX\bbetahat \right)^\top
\left(\by - \bX\bbetahat \right) + \frac{\lambda}{2} \sum_{d = 1}^D\hat{\beta}_d ^2.
\end{align*}
$$


Here, $\lambda$ is a *tuning parameter* which represents the amount of regularization. A large $\lambda$ means a greater penalty on the $\bbetahat$ estimates, meaning more shrinkage of these estimates toward 0. $\lambda$ is not estimated by the model but rather chosen before fitting, typically through {doc}`cross validation </content/appendix/methods>`. 



```{note}
Note that the Ridge loss function does *not* penalize the magnitude of the intercept estimate, $\hat{\beta}_0$. Intuitively, a greater intercept does not suggest overfitting. 
```



As in ordinary linear regression, we start estimating $\bbetahat$ by taking the derivative of the loss function. First note that since $\hat{\beta}_0$ is not penalized, 


$$
\begin{align*}
\frac{\partial}{\partial \bbetahat} \left( \frac{\lambda}{2}\sum_{d = 1}^D \hat{\beta}_d^2\right) &= \begin{bmatrix} 0 \\ \lambda \hat{\beta}_1 \\ ... \\ \lambda \hat{\beta}_D \end{bmatrix} \\
&= \lambda I' \bbetahat,
\end{align*}
$$


where $I'$ is the identity matrix of size $D+1$ *except* the first element is a 0. Then, adding in the derivative of the RSS discussed in {doc}`chapter 1 </content/c1/s1/loss_minimization>`, we get


$$
\begin{align*}
\dadb{\mathcal{L}_\text{Ridge}(\bbetahat)}{\bbetahat} &= - \bX^\top\left( \by - \bX \bbetahat \right) + \lambda I'\bbetahat.
\end{align*}
$$


Setting this equal to 0 and solving for $\bbetahat$, we get our estimates:


$$
\begin{align*}
\bbetahat\left(\bX^\top \bX + \lambda I' \right) &= \bX^\top \by 
\\
\bbetahat &= \left(\bX^\top \bX + \lambda I' \right)^{-1} \bX^\top \by,
\end{align*}
$$


## Lasso Regression

Lasso regression differs from Ridge regression in that its loss function uses the L1 norm for the $\bbetahat$ estimates rather than the L2 norm. This means we penalize the sum of absolute values of the $\hat{\beta}$s, rather than the sum of their squares. 


$$
\begin{align*}
\mathcal{L}_\text{Lasso}(\bbetahat) &= \frac{1}{2}\left(\by - \bX\bbetahat \right)^\top
\left(\by - \bX\bbetahat \right) + \lambda \sum_{d = 1}^D|\beta_d|.
\end{align*}
$$


As usual, let's then calculate the gradient of the loss function with respect to $\bbetahat$:


$$
\dadb{\mathcal{L}(\bbetahat)}{\bbetahat} = - \bX^\top\left( \by - \bX \bbetahat \right) + \lambda I'\text{ sign}(\bbetahat),
$$


where again we use $I'$ rather than $I$ since the magnitude of the intercept estimate $\hat{\beta}_0$ is not penalized. 



Unfortunately, we cannot find a closed-form solution for the $\bbetahat$ that minimize the Lasso loss. Numerous methods exist for estimating the $\bbetahat$, though using the gradient calculated above we could easily reach an estimate through {doc}`gradient descent </content/appendix/methods>`. The {doc}`construction </content/c2/s2/regularized>` in the next section uses this approach. 

