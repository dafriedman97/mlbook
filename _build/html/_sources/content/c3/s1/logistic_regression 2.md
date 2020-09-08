# Logistic Regression

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



In linear regression, we modeled our target variable as a linear combination of the predictors plus a random error term. This meant that the fitted value could be any real number. Since our target in classification is not any real number, the same approach wouldn't make sense in this context. Instead, logistic regression models a *function* of the target variable as a linear combination of the predictors, then converts this function into a fitted value in the desired range. 



## Binary Logistic Regression



### Model Structure

In the binary case, we denote our target variable with $Y_n \in \{0, 1\}$. Let $p_n = P(Y_n = 1)$ be our estimate of the probability that $Y_n$ is in class 1. We want a way to express $p_n$ as a function of the predictors ($\bx_n$) that is between 0 and 1. Consider the following function, called the *log-odds* of $p_n$. 


$$
f(p_n) = \log\left(\frac{p_n}{1-p_n}\right).
$$


Note that its domain is $(0, 1)$ and its range is all real numbers. This suggests that modeling the log-odds as a linear combination of the predictors—resulting in $f(p_n) \in \R$—would correspond to modeling $p_n$ as a value between 0 and 1. This is exactly what logistic regression does. Specifically, it assumes the following structure. 


$$
\begin{align*}
f(\hat{p}_n) = \log\left(\frac{\hat{p}_n}{1-\hat{p}_n}\right) &= \hat{\beta}_0 + \hat{\beta}_1 x_{n1} + \dots + \hat{\beta}_Dx_{nD} \\
&= \bbetahat^\top \bx_n.
\end{align*}
$$


```{admonition} Math Note
The *logistic function* is a common function in statistics and machine learning. The logistic function of $z$, written as $\sigma(z)$, is given by 

$$
\sigma(z) = \frac{1}{1 + \exp(-z)}.
$$

The derivative of the logistic function is quite nice. 

$$
\sigma'(z) = \frac{0 +\exp(-z)}{(1 + \exp(-z))^2} = \frac{1}{1 + \exp(-z)}\cdot\frac{\exp(-z)}{1 + \exp(-z)} = \sigma(z)(1-\sigma(z)).
$$
```



Ultimately, we are interested in $\hat{p}_n$, not the log-odds $f(\hat{p}_n)$. Rearranging the log-odds expression, we find that $\hat{p}_n$ is the logistic function of $\bbetahat^\top \bx_n$ (see the *Math Note* above for information on the logistic function). That is,


$$
\hat{p}_n = \sigma(\bbetahat^\top \bx_n) = \frac{1}{1 + \exp(-\bbetahat^\top \bx_n)}.
$$


By the derivative of the logistic function, this also implies that 


$$
\dadb{\hat{p}_n}{\bbetahat} = \dadb{\sigma(\bbetahat^\top\bx_n)}{\bbetahat} = \sigma(\bbetahat^\top \bx_n)\left(1-\sigma(\bbetahat^\top\bx_n) \right)\cdot \bx_n
$$


### Parameter Estimation 

We will estimate $\bbetahat$ with maximum likelihood. The PMF for $Y_n \sim \text{Bern}(p_n)$ is given by


$$
p(y_n) = p_n^{y_n}(1-p_n)^{1-y_n} =\sigma(\bbeta^\top \bx_n) ^{y_n} \left(1-\sigma(\bbeta^\top\bx_n) \right)^{1-y_n}.
$$


Notice that this gives us the correct probability for $y_n = 0$ and $y_n = 1$. 

Now assume we observe the target variables for our training data, meaning $Y_1, \dots, Y_n$ crystalize into $y_1, \dots, y_n$. We can write the likelihood and log-likelihood. 


$$
\begin{align*}
L(\bbeta; \{y_n, \bx_n\}_{n = 1}^N) &= \prodN p(y_n) \\
&= \prodN \sigma(\bbeta^\top\bx_n) ^{y_n} \left(1-\sigma(\bbeta^\top\bx_n)\right)^{1-y_n}
\\
\log L(\bbeta; \{y_n, \bx_n\}_{n = 1}^N)  &= \sumN y_n\log \sigma(\bbeta^\top\bx_n) + (1-y_n)\log\left( 1- \sigma(\bbeta^\top\bx_n)\right) 
\end{align*}
$$


Next, we want to find the values of $\bbetahat$ that maximize this log-likelihood. Using the derivative of the logistic function for $\bbeta^\top \bx_n$ discussed above, we get


$$
\begin{align*}
\dadb{\log L(\bbeta; \{\by_n, \bx_n\}_{n = 1}^N)}{\bbeta} &=\sumN y_n\frac{1}{\sigma(\bbeta^\top \bx_n)}\cdot\dadb{\sigma(\bbeta^\top\bx_n)}{\bbeta} - (1-y_n)\frac{1}{1-\sigma(\bbeta^\top\bx_n)}\cdot\dadb{\sigma(\bbeta^\top\bx_n)}{\bbeta}
\\
&= \sumN y_n\left(1-\sigma(\bbeta^\top\bx_n) \right)\cdot \bx_n - (1-y_n)\sigma(\bbeta^\top\bx_n)\cdot \bx_n
\\
&= \sumN y_n\bx_n - \sigma(\bbeta^\top\bx_n)\bx_n \\
&= \sumN (y_n - p_n)\bx_n.
\end{align*}
$$


Next, let $\mathbf{p} = \begin{pmatrix} p_1 & p_2 & \dots & p_N \end{pmatrix}^\top$be the vector of probabilities. Then we can write this derivative in matrix form as 


$$
\dadb{\log L(\bbeta; \{y_n, \bx_n\}_{n = 1}^N)}{\bbeta} = \bX^T (\by - \mathbf{p}).
$$

Ideally, we would find $\bbetahat$ by setting this gradient equal to 0 and solving for $\bbeta$. Unfortunately, there is no closed form solution. Instead, we can estimate $\bbetahat$ through gradient descent using the derivative above. Note that gradient descent minimizes a loss function, rather than maximizing a likelihood function. To get a loss function, we would simply take the negative log-likelihood. Alternatively, we could do gradient *ascent* on the log-likelihood. 



## Multiclass Logistic Regression

Multiclass logistic regression generalizes the binary case into the case where there are three or more possible classes. 



### Notation 

First, let's establish some notation. Suppose there are $K$ classes total. When $y_n$ can fall into three or more classes, it is best to write it as a *one-hot vector*: a vector of all zeros and a single one, with the location of the one indicating the variable's value. For instance, 


$$
\by_n = \begin{bmatrix} 0 \\ 1 \\ ... \\ 0 \end{bmatrix} \in \mathbb{R}^K
$$


indicates that the $n^\text{th}$ observation belongs to the second of $K$ classes. Similarly, let $\hat{\mathbf{p}}_n$ be a vector of estimated probabilities for observation $n$, where the $j^\text{th}$ entry indicates the probability that observation $n$ belongs to class $j$. Note that this vector must be non-negative and add to 1. For the example above,


$$
\hat{\mathbf{p}}_n = \begin{bmatrix} 0.01 \\ 0.98 \\ ... \\ 0.00 \end{bmatrix} \in \mathbb{R}^K
$$
would be a pretty good estimate. 



Finally, we need to write the coefficients for each class. Suppose we have $D$ predictor variables, including the intercept (i.e. $\bx_n \in \mathbb{R}^D$ where the first term in $\bx_n$ is an appended 1). We can let $\bbetahat_k$ be the length-$D$ vector of coefficient estimates for class $k$. Alternatively, we can use the matrix 


$$
\hat{\textbf{B}} = \begin{bmatrix} \bbetahat_1 & \dots & \bbetahat_K \end{bmatrix} \in \mathbb{R}^{D \times K},
$$


to jointly represent the coefficients of all classes.



### Model Structure

Let's start by defining $\hat{\mathbf{z}}_n$ as 


$$
\hat{\mathbf{z}}_n = \hat{\mathbf{B}}^\top \mathbf{x}_n \in \mathbb{R}^K.
$$


Note that $\hat{\mathbf{z}}_n$ has one entry per class. It seems we might be able to fit $\hat{\mathbf{B}}$ such that the $k^\text{th}$ element of $\hat{\mathbf{z}}_n$ gives $P(\by_n = k)$. However, it would be difficult to at the same time ensure the entries in $\hat{\mathbf{z}}_n$ sum to 1. Instead, we apply a *softmax* transformation to $\hat{\mathbf{z}}_n$ in order to get our estimated probabilities. 



```{admonition} Math Note
For some length-$K$ vector $\mathbf{z}$ and entry $k$, the *softmax* function is given by

$$
\text{softmax}_k(\mathbf{z}) = \frac{\exp(z_k)}{\sum_{j = 1}^K \exp(z_j)}.
$$

Intuitively, if the $k^\text{th}$ entry of $\mathbf{z}$ is large relative to the others, $\text{softmax}_k(\mathbf{z})$ will be as well. 

If we drop the $k$ from the subscript, the softmax is applied over the entire vector. I.e.,

$$
\text{softmax}(\mathbf{z}) = \begin{bmatrix} \text{softmax}_1(\mathbf{z}) & \dots & \text{softmax}_K(\mathbf{z}) \end{bmatrix}^\top
$$

```



To obtain a valid set of probability estimates for $\hat{\mathbf{p}}_n$, we apply the softmax function to $\hat{\mathbf{z}}_n$. That is,


$$
\hat{\mathbf{p}}_n = \text{softmax}(\hat{\mathbf{z}}_n) = \text{softmax}(\hat{\mathbf{B}}^\top \mathbf{x}_n).
$$

Let $\hat{p}_{nk}$, the $k^\text{th}$ entry in $\hat{\mathbf{p}}_n$ give the probability that observation $n$ is in class $k$. 



### Parameter Estimation

Now let's see how the estimates in $\hat{\mathbf{B}}$ are actually fit. 

#### The Likelihood Function

As in binary logistic regression, we estimate $\hat{\mathbf{B}}$ by maximizing the (log) likelihood. Let $I_{nk}$ be an indicator that equals 1 if observation $n$ is in class $k$ and 0 otherwise. The likelihood and log-likelihood are 


$$
\begin{align*}
L(\mathbf{B}; \{\by_n, \bx_n\}_{n = 1}^N) &= \prodN \prod_{k = 1}^K p_{nk}^{I_{nk}}
\\
\log L(\mathbf{B}; \{\by_n, \bx_n\}_{n = 1}^N) &= \sumN \sum_{k = 1}^K I_{nk} \log p_{nk}  \\
&= \sumN \sum_{k = 1}^K I_{nk} \log\left(\text{sigmoid}_k(\mathbf{z}_n)\right)\\
&= \sumN \sum_{k = 1}^K I_{nk}\left(z_{nk} - \log\left(\sum_{i = 1}^K \exp(z_{ni})\right) \right),
\end{align*}
$$


where the last equality comes from the fact that


$$
\log(\text{sigmoid}_k(\mathbf{z}_n)) = \log\left(\frac{\exp(z_{nk})}{\sum_{j = 1}^K \exp(z_{nk})}\right) = z_{nk} - \log\left(\sum_{j = 1}^K \exp(z_{nj})\right).
$$



#### The Derivative

Now let's look at the derivative. Specifically, let's look at the derivative of the log-likelihood with respect to the coefficients from the $j^\text{th}$ class, $\bbeta_j$. Note that 


$$
\begin{cases}
\dadb{z_{nk}}{\bbeta_j} = \bx_n, \hspace{1cm} & j = k  \\
\dadb{z_{nk}}{\bbeta_j} = 0,\hspace{1cm} &\text{otherwise}.  \\
\end{cases}
$$
This implies that 


$$
\dadb{}{\bbeta_j}\sum_{k = 1}^K I_{nk} z_{nk}  = I_{nj}\bx_n,
$$


since the derivative is automatically 0 for all terms but the $j^\text{th}$ and $\bx_n$ if $I_{nj} = 1$. Then,


$$
\begin{align*}
\dadb{}{\bbeta_j}\log L(\mathbf{B}; \{\by_n, \bx_n\}_{n = 1}^N) &= \sumN  \left( I_{nj}\bx_n - \sum_{k = 1}^K I_{nk}\frac{\exp(z_{nj})\bx_n}{\sum_{i = 1}^K \exp(z_{ni})} \right) \\
&= \sumN \left(I_{nj} - \sum_{k = 1}^KI_{nk}\text{softmax}_j(\mathbf{z}_n) \right)\bx_n \\
&= \sumN \left(I_{nj} - p_{nj}\sum_{k = 1}^KI_{nk} \right)\bx_n \\
&= \sum_{n = 1}^N (I_{nj} - p_{nj})\bx_n. 
\end{align*}
$$


In the last step, we drop the $\sum_{k = 1}^K I_{nk}$ since this must equal 1. This gives us the gradient of the loss function with respect to a given class's coefficients, which is enough to build our model. It is possible, however, to simplify these expressions further, which is useful for gradient descent. These simplifications are given below. 



#### Simplifying

This gradient above can also be written more compactly in matrix format. Let


$$
\mathbf{i}_j = \begin{bmatrix} I_{1j} \\ ... \\ I_{nj} \end{bmatrix}, \hspace{.25cm} \mathbf{p}'_j =  \begin{bmatrix} p_{1j} \\ ... \\ p_{nj} \end{bmatrix}
$$


identify whether each observation was in class $j$ and give the probability that the observation is in class $j$, respectively.

```{note}
Note that we use $\mathbf{p}'$ rather than $\mathbf{p}$ since $\mathbf{p}_n$ was used to represent the probability that observation $n$ belonged to a series of classes while $\mathbf{p}'_j$ refers to the probability that a series of observations belong to class $j$. 
```

 Then, we can write 


$$
\dadb{}{\bbeta_j}\log L(\mathbf{B}; \{\by_n, \bx_n\}_{n = 1}^N)  = \bX^\top (\mathbf{i}_j - \mathbf{p}'_j).
$$


Further, we can simultaneously represent the derivative of the loss function with respect to *each* of the class's coefficients. Let 


$$
\mathbf{I} = \begin{bmatrix} \mathbf{i}_1 & \dots & \mathbf{i}_K \end{bmatrix} \in \mathbb{R}^{N \times K}, \hspace{.25cm} \mathbf{P} = \begin{bmatrix} \mathbf{p}'_1 & \dots & \mathbf{p}'_K \end{bmatrix} \in \mathbb{R}^{N \times K}.
$$


We can then write 


$$
\dadb{}{\mathbf{B}}\log L(\mathbf{B}; \{\by_n, \bx_n\}_{n = 1}^N)  = \mathbf{X}^\top \left( \mathbf{I} - \mathbf{P}\right) \in \mathbb{R}^{D \times K}.
$$


Finally, we can also write $\hat{\mathbf{P}}$ (the estimate of $\mathbf{P}$) as a matrix product, which will make calculations more efficient. Let


$$
\hat{\mathbf{Z}} = \bX \hat{\mathbf{B}} \in \mathbb{R}^{N \times K},
$$


where the $n^\text{th}$ row is equal to $\hat{\mathbf{z}}_n$. Then, 


$$
\hat{\mathbf{P}} = \text{softmax}(\hat{\mathbf{Z}}) \in \mathbb{R}^{N \times K},
$$


where the softmax function is applied to each row.
