Bayesian Regression
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
\newcommand{\bT}{\mathbf{T}}
\newcommand{\dadb}[2]{\frac{\partial #1}{\partial #2}}
\newcommand{\iid}{\overset{\small{\text{i.i.d.}}}{\sim}}
$$

In the Bayesian approach to statistical inference, we treat our parameters as random variables and assign them a prior distribution. This forces our estimates to reconcile our existing beliefs about these parameters with new information given by the data. This approach can be applied to linear regression by assigning the regression coefficients a prior distribution. 



We also may wish to perform Bayesian regression not because of a prior belief about the coefficients but in order to minimize model complexity. By assigning the parameters a prior distribution with mean 0, we force the posterior estimates to be closer to 0 than they would otherwise. This is a form of regularization similar to the Ridge and Lasso methods discussed in the {doc}`previous section <regularized>`. 



## The Bayesian Structure

To demonstrate Bayesian regression, we'll follow three typical steps to Bayesian analysis: writing the likelihood, writing the prior density, and using Bayes' Rule to get the posterior density. In the {ref}`results <results>` below, we use the posterior density to calculate the maximum-a-posteriori (MAP)—the equivalent of calculating the $\hat{\bbeta}$ estimates in ordinary linear regression. 



### 1. The Likelihood

As in the typical regression set-up, let's assume 


$$
Y_n \iid \mathcal{N}\left(\bbeta^\top \bx_n , \sigma^2\right).
$$


We can write the collection of observations jointly as  


$$
\begin{align*}
\by &\sim \mathcal{N}\left( \bX\bbeta, \bSigma\right),
\end{align*}
$$


where $\by \in \mathbb{R}^N$ and $\bSigma = \sigma^2 I_N \in \mathbb{R}^{N \times N}$ for some *known* scalar $\sigma^2$. Note that $\by$ is a vector of random variables—it is not capitalized in order to distinguish it from a matrix.



```{note}
See [this lecture](https://www.statlect.com/fundamentals-of-statistics/Bayesian-regression) for an example of Bayesian regression without the assumption of known variance. 
```



We can then get our likelihood and log-likelihood using the Multivariate Normal. 


$$
\begin{align*}
L(\bbeta; \bX, \by) 
&= 
\frac{1}{\sqrt{(2\pi)^N|\bSigma|}}\exp\left(-\frac{1}{2}(\by - \bX\bbeta)^\top\bSigma^{-1}(\by - \bX\bbeta) \right) 
\\
&\propto \exp\left(-\frac{1}{2}(\by - \bX\bbeta)^\top\bSigma^{-1}(\by - \bX\bbeta) \right) 
\\
\log L(\bbeta; \bX, \by) &= -\frac{1}{2}(\by - \bX\bbeta)^\top\bSigma^{-1}(\by - \bX\bbeta).
\end{align*}
$$



### 2. The Prior

Now, let's assign $\bbeta$ a prior distribution. We typically assume 

$$
\bbeta \sim \mathcal{N}(\mathbf{0}, \bT),
$$


where $\bbeta \in \mathbb{R}^D$ and $\bT = \tau I_D \in \mathbb{R}^{D \times D}$ for some scalar $\tau$. We choose $\tau$ (and therefore $\bT$) ourselves, with a greater $\tau$ giving less weight to the prior. 



The prior density is given by


$$
\begin{align*}
p(\bbeta) &= 
\frac{1}{\sqrt{(2\pi)^D|\bT|}}\exp\left(-\frac{1}{2}\bbeta^\top\bT^{-1}\bbeta \right) 
\\
&\propto \exp\left(-\frac{1}{2}\bbeta^\top\bT^{-1}\bbeta \right)
\\
\log p(\bbeta) &= -\frac{1}{2}\bbeta^\top \bT^{-1}\bbeta.
\end{align*}
$$



### 3. The Posterior



We are then interested in a posterior density of $\bbeta$ given the data, $\bX$ and $\by$.

Bayes' rule tells us that the posterior density of the coefficients is proportional to the likelihood of the data times the prior density of the coefficients. Using the two previous results, we have


$$
\begin{align*}
p(\bbeta|\bX, \by) &\propto L(\bbeta; \bX, \by) p(\bbeta) 
\\
\log p(\bbeta|\bX, \by) &= \log L(\bbeta; \bX, \by) + \log p(\bbeta) + k
\\
&=  -\frac{1}{2}(\by - \bX\bbeta)^\top\bSigma^{-1}(\by - \bX\bbeta) - \frac{1}{2}\bbeta^\top \bT^{-1}\bbeta + k 
\\
&= -\frac{1}{2\sigma^2}(\by - \bX\bbeta)^\top(\by - \bX\bbeta) - \frac{1}{2\tau}\bbeta^\top \bbeta + k 
\end{align*}
$$


where $k$ is some constant that we don't care about.



(results)= 

## Results

### Intuition

Often in the Bayesian setting it is infeasible to obtain the entire posterior distribution. Instead, one typically looks at the maximum-a-posteriori (MAP), the value of the parameters that maximize the posterior density. In our case, the MAP is the $\bbetahat$ that maximizes 


$$
\begin{align*}
\log p(\bbetahat|\bX, \by) &= -\frac{1}{2\sigma^2}(\by - \bX\bbetahat)^\top(\by - \bX\bbetahat) - \frac{1}{2\tau}\bbetahat^\top \bbetahat.
\end{align*}
$$


This is equivalent to finding the $\bbetahat$ that minimizes the following loss function, where $\lambda = 1/\tau$. 


$$
\begin{align}
L(\bbetahat) &= \frac{1}{2}(\by - \bX\bbetahat)^\top(\by - \bX\bbetahat) + \frac{\lambda}{2}\bbetahat^\top \bbetahat 
\\
&= \frac{1}{2}(\by - \bX\bbetahat)^\top(\by - \bX\bbetahat) + \frac{\lambda}{2} \sum_{d = 0}^D\hat{\beta}_d.
\end{align}
$$


Notice that this is extremely close to the Ridge loss function discussed in the {doc}`previous section <regularized>`—it is not quite equal to the Ridge loss function since it also penalizes the magnitude of the intercept, though this difference could be eliminated by changing the prior distribution of the intercept.



This shows that Bayesian regression with a mean-zero Normal prior distribution is essentially equivalent to Ridge regression. Decreasing $\tau$, just like increasing $\lambda$, increases the amount of regularization. 



### Full Results

Now let's actually derive the MAP by calculating the gradient of the log posterior density. 



```{admonition} Math Note
For a symmetric matrix $\mathbf{W}$, 

$$
\frac{\partial}{\partial \mathbf{s}}\left(\mathbf{q} - \mathbf{A}\mathbf{s} \right)^\top \mathbf{W}\left(\mathbf{q} - \mathbf{A}\mathbf{s}\right) = -2\mathbf{A}^\top \mathbf{W}\left(\mathbf{q} - \mathbf{A}\mathbf{s}\right)
$$

This implies that

$$
\frac{\partial}{\partial \mathbf{s}}\mathbf{s}^\top \mathbf{W}\mathbf{s} = 
\frac{\partial}{\partial \mathbf{s}} (\mathbf{0} - I\mathbf{s})^\top \mathbf{W} (\mathbf{0} - I\mathbf{s})= 
2\mathbf{W}\mathbf{s}.
$$
```



Using the *Math Note* above, we have


$$
\begin{align*}
\log p(\bbetahat|\bX, \by) &=  -\frac{1}{2}(\by - \bX\bbeta)^\top\bSigma^{-1}(\by - \bX\bbeta) - \frac{1}{2}\bbeta^\top \bT^{-1}\bbeta \\
\dadb{}{\bbeta} \log p(\bbeta|\bX, \by) &= \bX^\top \bSigma^{-1}(\by - \bX \bbeta) - \bT^{-1}\bbeta.
\end{align*}
$$


We calculate the MAP by setting this gradient equal to 0:


$$
\begin{align*}
\bbetahat &= \left(\bX^\top\bSigma^{-1} \bX + \bT^{-1}\right)^{-1}\bX^\top\bSigma^{-1}\by \\
&= \left(\frac{1}{\sigma^2}\bX^\top\bX + \frac{1}{\tau} I\right)^{-1}\frac{1}{\sigma^2}\bX^\top\by.
\end{align*}
$$