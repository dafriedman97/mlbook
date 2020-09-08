GLMs
==============

$$
\newcommand{\sumN}{\sum_{n = 1}^N}
\newcommand{\sumn}{\sum_n}
\newcommand{\prodN}{\prod_{n = 1}^N}
\newcommand{\by}{\mathbf{y}} \newcommand{\bX}{\mathbf{X}}
\newcommand{\bx}{\mathbf{x}}
\newcommand{\bbeta}{\boldsymbol{\beta}}
\newcommand{\btheta}{\boldsymbol{\theta}}
\newcommand{\blambda}{\boldsymbol{\lambda}}
\newcommand{\bbetahat}{\boldsymbol{\hat{\beta}}}
\newcommand{\bthetahat}{\boldsymbol{\hat{\theta}}}
\newcommand{\bSigma}{\boldsymbol{\Sigma}}
\newcommand{\bT}{\mathbf{T}}
\newcommand{\dadb}[2]{\frac{\partial #1}{\partial #2}}
\newcommand{\iid}{\overset{\small{\text{i.i.d.}}}{\sim}}
$$

Ordinary linear regression comes with several assumptions that can be relaxed with a more flexible model class: generalized linear models (GLMs). Specifically, OLS assumes

1. The target variable is a linear function of the input variables
2. The errors are Normally distributed
3. The variance of the errors is constant

When these assumptions are violated, GLMs might be the answer.



## GLM Structure

A GLM consists of a *link function* and a *random component*. The *random component* identifies the distribution of the target variable $y_n$ conditional on the input variables $\bx_n$. For instance, we might model $Y_n$ as a Poisson random variable where the rate parameter $\lambda_n$ depends on $\bx_n$.

The *link function* specifies how $\bx_n$ relates to the expected value of the target variable, $\mu_n = E(Y_n)$. Let $\eta$ be a linear function of the input variables, i.e. $\eta_n = \bbeta^\top \bx_n$ for some coefficients $\bbeta$. We then chose a nonlinear link function to relate $\mu$ to $\eta$. For link function $g$ we have 


$$
\eta_n = g(\mu_n).
$$


In a GLM, we calculate $\eta$ *before* calculating $\mu$, so we often work with the inverse of $g$:


$$
\mu_n = g^{-1}(\eta_n)
$$


```{note}
Note that because $\eta_n$ is a function of the data, it will vary for each observation (though the $\beta$s will not). 
```



In total then, a GLM assumes 


$$
\begin{aligned}
Y_n &\sim F_{\mu_n} \\
\mu_n &= g^{-1}(\eta_n) \\
\eta_n &= \bbeta^\top \bx_n,
\end{aligned}
$$


where $F$ is some distribution with mean parameter $\mu_n$. 



## Fitting a GLM

"Fitting" a GLM, like fitting ordinary linear regression, really consists of estimating the coefficients, $\bbeta$. Once we know $\bbeta$, we have $\eta$. Once we have a link function, $\eta$ gives us $\mu$ through $g^{-1}$. A GLM can be fit in these four steps:



1. Specify the distribution of $Y_n$, indexed by its mean parameter $\mu_n$. 

2. Specify the link function $\eta_n = g(\mu_n)$. 

3. Identify a loss function. This is typically the negative log-likelihood. 

4. Find the $\bbetahat$ that minimize that loss function.

   

In general, we can write the log-likelihood across our observations for a GLM as follows. 


$$
\log L\left(\{\mu_n\}_{n = 1}^N; \{Y_n\}_{n = 1}^N\right) = \sumN \log L(\mu_n; Y_n) = \sumN \log L(g^{-1}(\eta_n); Y_n) = \sumN \log L(g^{-1}(\bbeta^\top \bx_n); Y_n).
$$


This shows how the log-likelihood depends on $\bbeta$, the parameters we want to estimate. To fit the GLM, we want to find the $\bbetahat$ to maximize this log-likelihood. 



## Example: Poisson Regression

<u>Step 1</u>

Suppose we choose to model $Y_n$ conditional on $\bx_n$ as a Poisson random variable with rate parameter $\lambda_n$:


$$
Y_n|\bx_n \sim \text{Pois}(\lambda_n).
$$

Since the expected value of a Poisson random variable is its rate parameter, $E(Y_n) = \mu_n = \lambda_n$. 

<u>Step 2</u>

To determine the link function, let's think in terms of its inverse, $\lambda_n = g^{-1}(\eta_n)$.  We know that $\lambda_n$ must be non-negative and $\eta_n$ could be anywhere in the reals since it is a linear function of $\bx_n$. One function that works is 


$$
\lambda_n = \exp(\eta_n),
$$


meaning 


$$
\eta_n = g(\lambda_n) = \log(\lambda_n). 
$$


This is the "canonical link" function for Poisson regression. More on that [here](https://en.wikipedia.org/wiki/Generalized_linear_model#Link_function). 



<u>Step 3</u>

Let's derive the negative log-likelihood for the Poisson. Let $\blambda = \begin{bmatrix} \lambda_1, \dots, \lambda_N\end{bmatrix}^\top$. 



```{admonition} Math Note
The PMF for $Z \sim \text{Pois}(\lambda)$ is 

$$
p(z) = \frac{e^{-\lambda}\lambda^z}{z!} \propto e^{-\lambda}\lambda^z.
$$
```




$$
\begin{align*}
L(\blambda; \{Y_n\}_{n = 1}^N) &= \prodN \exp(-\lambda_n) \lambda_n^{Y_n}
\\
\log L(\blambda; \{Y_n\}_{n = 1}^N) &= \sumN Y_n\log\lambda_n - \lambda_n. 
\end{align*}
$$


Now let's get our loss function, the negative log-likelihood. Recall that this should be in terms of $\bbeta$ rather than $\blambda$ since $\bbeta$ is what we control. 


$$
\begin{align}
\mathcal{L}_N(\bbeta) &= -\left(\sumN Y_n\log(\exp(\eta_n)) - \exp(\eta_n)\right)
\\
&= \sumN \left(\exp(\eta_n) - Y_n \eta_n \right)
\\
&= \sumN \left(\exp(\bbeta^\top \bx_n) - Y_n\bbeta^\top\bx_n\right).
\end{align}
$$


<u>Step 4</u>

We obtain $\bbetahat$ by minimizing this loss function. Let's take the derivative of the loss function with respect to $\bbeta$. 


$$
\dadb{\mathcal{L}_N(\bbeta)}{\bbeta} = \sumN\left(\exp(\bbeta^\top\bx_n)\bx_n - y_n\bx_n \right).
$$


Ideally, we would solve for $\bbetahat$ by setting this gradient equal to 0. Unfortunately, there is no closed-form solution. Instead, we can approximate $\bbetahat$ through {doc}`gradient descent </content/appendix/methods>`. This is done in the {doc}`construction </content/c2/s2/GLMs>` section. 



Since gradient descent calculates this gradient a large number of times, it's important to calculate it efficiently. Let's see if we can clean this expression up. First recall that
$$
\hat{y}_n = \hat{\lambda}_n = \exp(\bbetahat^\top \bx_n).
$$

The loss function can then be written as 


$$
\dadb{\mathcal{L}_N(\bbetahat)}{\bbetahat} = \sumN\left(\hat{y}_n\bx_n - y_n\bx_n \right).
$$

Further, this can be written in matrix form as 

$$
\dadb{\mathcal{L}_N(\bbetahat)}{\bbetahat} = \bX^\top(\hat{\by} - \by),
$$


where $\hat{\by}$ is the vector of fitted values. Finally note that this vector can be calculated as 

$$
\hat{\by} = \exp(\bX \bbetahat),
$$

where the exponential function is applied element-wise to each observation.



____

Many other GLMs exist. One important example is logistic regression, the topic of the next chapter.