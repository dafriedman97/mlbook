# Approach 1: Minimizing Loss

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

## 1. Simple Linear Regression



### Model Structure

*Simple linear regression* models the target variable, $y$, as a linear function of just one predictor variable, $x$, plus an error term, $\epsilon$. We can write the entire model for the $n^\text{th}$ observation as 

$$
\begin{align*}
y_n &= \beta_0 + \beta_1 x_n + \epsilon_n.
\end{align*}
$$



Fitting the model then consists of estimating two parameters: $\beta_0$ and $\beta_1$. We call our estimates of these parameters $\hat{\beta}_0$ and $\hat{\beta}_1$, respectively. Once we've made these estimates, we can form our prediction for any given $x_n$ with 

$$
\hat{y}_n = \hat{\beta}_0 + \hat{\beta}_1 x_n. 
$$

One way to find these estimates is by minimizing a loss function. Typically, this loss function is the *residual sum of squares* (RSS). The RSS is calculated with


$$
\begin{align*}
\mathcal{L}(\hat{\beta}_0, \hat{\beta}_1) &= \frac{1}{2}\sumN \left(y_n - \hat{y}_n\right)^2.
\end{align*}
$$



We divide the sum of squared errors by 2 in order to simplify the math, as shown below. Note that doing this does not affect our estimates because it does not affect which $\hat{\beta}_0$ and $\hat{\beta}_1$ minimize the RSS.



### Parameter Estimation

Having chosen a loss function, we are ready to derive our estimates. First, let's rewrite the RSS in terms of the estimates:


$$
\mathcal{L}(\hat{\beta}_0, \hat{\beta}_1) = \frac{1}{2}\sumN\left(y_n - \left(\hat{\beta}_0 + \hat{\beta}_1x_n\right)\right)^2.
$$


To find the intercept estimate, start by taking the derivative of the RSS with respect to $\hat{\beta}_0$:


$$
\begin{align}
\dadb{\mathcal{L}(\hat{\beta}_0, \hat{\beta}_1)}{\hat{\beta}_0} &= -\sumN \left(y_n - \hat{\beta}_0 - \hat{\beta}_1x_n\right)  \\
&= -N(\bar{y} - \hat{\beta}_0 - \hat{\beta}_1\bar{x}),
\end{align}
$$


where $\bar{y}$ and $\bar{x}$ are the sample means. Then set that derivative equal to 0 and solve for $\hat{\beta}_0$:

$$
\begin{align*}
\hat{\beta}_0 &= \bar{y} - \hat{\beta}_1\bar{x}.
\end{align*}
$$

This gives our intercept estimate, $\hat{\beta}_0$, in terms of the slope estimate, $\hat{\beta}_1$. To find the slope estimate, again start by taking the derivative of the RSS: 


$$
\begin{align}
\dadb{\mathcal{L}(\hat{\beta}_0, \hat{\beta}_1)}{\hat{\beta}_1} &= - \sumN \left(y_n - \hat{\beta}_0 - \hat{\beta}_1 x_n\right)x_n.
\end{align}
$$


Setting this equal to 0 and substituting for $\hat{\beta}_0$, we get

$$
\begin{align*}
\sumN \left(y_n - (\bar{y} - \hat{\beta}_1 \bar{x}) - \hat{\beta}_1 x_n\right)x_n &= 0
\\
\hat{\beta}_1\sumN (x_n-\bar{x})x_n &= \sumN \left(y_n - \bar{y}\right)x_n 
\\
\hat{\beta}_1 &= \frac{\sumN x_n(y_n - \bar{y})}{\sumN x_n(x_n - \bar{x})}.
\end{align*}
$$


To put this in a more standard form, we use a slight algebra trick. Note that 


$$
\sumN c(z_n - \bar{z}) = 0
$$


for any constant $c$ and any collection $z_1, \dots, z_N$ with sample mean $\bar{z}$ (this can easily be verified by expanding the sum). Since $\bar{x}$ is a constant, we can then subtract  $\sumN \bar{x}(y_n - \bar{y})$ from the numerator and $\sumN \bar{x}(x_n - \bar{x})$ from the denominator without affecting our slope estimate. Finally, we get


$$
\hat{\beta}_1 = \frac{\sumN (x_n - \bar{x})(y_n - \bar{y})}{\sumN(x_n - \bar{x})^2}.
$$



## 2. Multiple Regression



### Model Structure

In multiple regression, we assume our target variable to be a linear combination of *multiple* predictor variables. Letting $x_{nj}$ be the $j^\text{th}$ predictor for observation $n$, we can write the model as



$$
\begin{align*}
y_n &= \beta_0 + \beta_1x_{n1} + \dots + \beta_D x_{nD} + \epsilon_n.
\end{align*}
$$



Using the vectors $\bx_n$ and $\bbeta$ defined in the {doc}`previous section </content/c1/concept>`, this can be written more compactly as 


$$
y_n = \bbeta^\top\bx_n + \epsilon_n.
$$



Then define $\bbetahat$ the same way as $\bbeta$ except replace the parameters with their estimates. We again want to find the vector $\hat{\bbeta}$ that minimizes the RSS: 



$$
\mathcal{L}(\bbetahat) = \frac{1}{2}\sumN \left( y_n - \bbetahat^\top \bx_n\right)^2 = \frac{1}{2}\sumN(y_n - \hat{y}_n)^2,
$$



Minimizing this loss function is easier when working with matrices rather than sums. Define $\by$ and $\bX$ with


$$
\by = \begin{bmatrix}y_1 \\ \dots \\ y_N  \end{bmatrix} \in \mathbb{R}^{N}, \hspace{.25cm} \bX = \begin{bmatrix} \bx_1^\top \\ \dots \\ \bx_N^\top  \end{bmatrix} \in \mathbb{R}^{N \times (D+1)},
$$



which gives $\hat{\by} = \bX\bbetahat \in \mathbb{R}^N$. Then, we can equivalently write the loss function as


$$
\mathcal{L}(\bbetahat) = \frac{1}{2}(\by - \bX\bbetahat)^\top(\by - \bX\bbetahat).
$$


### Parameter Estimation

We can estimate the parameters in the same way as we did for simple linear regression, only this time calculating the derivative of the RSS with respect to the entire parameter vector. First, note the commonly-used matrix derivative below [^ref1]. 



```{admonition} Math Note
For a symmetric matrix $\mathbf{W}$,

$$
\frac{\partial}{\partial \mathbf{s}}\left(\mathbf{q} - \mathbf{A}\mathbf{s} \right)^\top \mathbf{W} \left(\mathbf{q} - \mathbf{A}\mathbf{s}\right) = -2\mathbf{A}^\top\mathbf{W}\left(\mathbf{q} - \mathbf{A}\mathbf{s}\right)
$$
```




Applying the result of the Math Note, we get the derivative of the RSS with respect to $\bbetahat$ (note that the identity matrix takes the place of $\mathbf{W}$):


$$
\begin{align*}
\mathcal{L}(\bbetahat) &= \frac{1}{2}(\by - \bX\bbetahat)^\top(\by - \bX\bbetahat) 
\\
\dadb{\mathcal{L}(\bbetahat)}{\bbetahat} &= - \bX^\top(\by - \bX\bbetahat).
\end{align*}
$$

We get our parameter estimates by setting this derivative equal to 0 and solving for $\bbetahat$:


$$
\begin{align}
(\bX^\top \bX)\bbetahat &= \bX^\top \by \\\
\bbetahat = (\bX^\top\bX)^{-1}\bX^\top \by
\end{align}
$$


[^ref1]: A helpful guide for matrix calculus is [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)

