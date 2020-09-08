# Concept

$$
\newcommand{\sumN}{\sum_{n = 1}^N}
\newcommand{\sumn}{\sum_n}
\newcommand{\sumK}{\sum_{k = 1}^K}
\newcommand{\sumk}{\sum_k}
\newcommand{\prodN}{\prod_{n = 1}^N}
\newcommand{\prodK}{\prod_{k = 1}^K}
\newcommand{\by}{\mathbf{y}} 
\newcommand{\bX}{\mathbf{X}}
\newcommand{\bx}{\mathbf{x}}
\newcommand{\bp}{\mathbf{p}}
\newcommand{\bbeta}{\boldsymbol{\beta}}
\newcommand{\btheta}{\boldsymbol{\theta}}
\newcommand{\bmu}{\boldsymbol{\mu}}
\newcommand{\bpi}{\boldsymbol{\pi}}
\newcommand{\bbetahat}{\boldsymbol{\hat{\beta}}}
\newcommand{\bthetahat}{\boldsymbol{\hat{\theta}}}
\newcommand{\bSigma}{\boldsymbol{\Sigma}}
\newcommand{\bT}{\mathbf{T}}
\newcommand{\dadb}[2]{\frac{\partial #1}{\partial #2}}
\newcommand{\iid}{\overset{\small{\text{i.i.d.}}}{\sim}}
\newcommand{\collection}{\{\bx_n, y_n\}_{n = 1}^N}
\newcommand{\l}{\left(}
\newcommand{\r}{\right)}
$$

Discriminative classifiers, as we saw in the previous chapter, model a target variable as a direct function of one or more predictors. Generative classifiers, the subject of this chapter, instead view the predictors as being generated according to their class—i.e., they see the predictors as a function of the target, rather than the other way around. They then use Bayes' rule to turn $P(\bx_n|Y_n = k)$ into $P(Y_n = k|\bx_n)$. 

In generative classifiers, we view both the target and the predictors as random variables. We will therefore refer to the target variable with $Y_n$, but in order to avoid confusing it with a matrix, we refer to the predictor vector with $\bx_n$. 

Generative models can be broken down into the three following steps. Suppose we have a classification task with $K$ unordered classes, represented by $k = 1, \dots, K$. 

1. Estimate the density of the predictors conditional on the target belonging to each class. I.e., estimate $p(\bx_n|Y_n = k)$ for $k = 1, \dots, K$. 
2. Estimate the prior probability that a target belongs to any given class. I.e., estimate $P(Y_n = k)$ for $k = 1, \dots, K$. This is also written as $p(Y_n)$. 
3. Using Bayes' rule, calculate the posterior probability that the target belongs to any given class. I.e., calculate  $p(Y_n = k|\bx_n) \propto p(\bx_n|Y_n = k)p(Y_n = k)$ for $k = 1, \dots, K$. 

We then classify observation $n$ as being from the class for which $P(Y_n = k|\bx_n)$ is greatest. In math, 


$$
\hat{Y}_n = \underset{k}{\text{arg max }} p(Y_n = k|\bx_n).
$$


Note that we do not need $p(\bx_n)$, which would be the denominator in the Bayes' rule formula, since it would be equal across classes.



```{note}
This chapter is oriented differently from the others. The main methods discussed—Linear Discriminant Analysis, Quadratic Discriminant Analysis, and Naive Bayes—share much of the same structure. Rather than introducing each individually, we describe them together and note (in section 2.2) how they differ.
```





## 1. Model Structure

A generative classifier models two sources of randomness. First, we assume that out of the $K$ possible classes, each observation belongs to class $k$ independently with probability $\pi_k$. In other words, letting $\bpi =\begin{bmatrix} \pi_1 & \dots & \pi_K\end{bmatrix}^\top \in \mathbb{R}^{K}$, we assume the prior 


$$
y_n \iid \text{Cat}(\bpi).
$$


See the math note below on the Categorical distribution. 

```{admonition} Math Note
A random variable which takes on one of $K$ discrete and unordered outcomes with probabilities $\pi_1, \dots, \pi_K$ follows the Categorical distribution with parameter $\bpi = \begin{bmatrix} \pi_1 & \dots & \pi_K \end{bmatrix}^\top$, written $\text{Cat}(\bpi)$. For instance, a single die roll is distributed $\text{Cat}(\bpi)$ for $\bpi = \begin{bmatrix} 1/6 \dots 1/6 \end{bmatrix}^\top$.

The density for $Y \sim \text{Cat}(\bp)$ is defined as

$$
\begin{align*}
P(Y = 1) &= p_1 \\
&... \\
P(Y = K) &= p_K.
\end{align*}
$$


This can be written more compactly as

$$
p(y) = \prod_{k = 1}^K p_k ^{I_k}
$$

where $I_k$ is an indicator that equals 1 if $y = k$ and 0 otherwise.
```



We then assume some distribution for $\mathbf{x}_n$ conditional on observation $n$'s class, $Y_n$. We typically assume all the $\bx_n$ come from the same *family* of distributions, though the parameters depend on their class. For instance, we might have 


$$
\begin{align*}
\bx_n|(Y_n = 1) &\sim \text{MVN}(\bmu_1, \bSigma_1), \\ &... \\
\bx_{n}|(Y_n = K)  &\sim \text{MVN}(\bmu_K, \bSigma_K),
\end{align*}
$$


though we wouldn't let one conditional distribution be Multivariate Normal and another be Multivariate $t$. Note that it is possible, however, for the individual variables within the random vector $\bx_n$ to follow different distributions. For instance, if $\bx_n = \begin{bmatrix} x_{n1} & x_{n2} \end{bmatrix}^\top$, we might have


$$
\begin{align*}
x_{n1}|(Y_n = k) &\sim \text{Bin}(n, p_k)
\\
x_{n2}|(Y_n = k) &\sim \mathcal{N}(\bmu_k, \bSigma_k) 
\end{align*}
$$


The machine learning task is to estimate the parameters of these models—$\bpi$ for $Y_n$ and whatever parameters might index the possible distributions of $\bx_n|Y_n$, in this case $\bmu_k$ and $\bSigma_k$ for $k = 1, \dots, K$. Once that's done, we can estimate $p(Y_n = k)$ and $p(\bx_n|Y_n = k)$ for each class and, through Bayes' rule, choose the class that maximizes $p(Y_n = k|\bx_n)$.



## 2. Parameter Estimation



### 2.1 Class Priors

Let's start by deriving the estimates for $\bpi$, the class priors. Let $I_{nk}$ be an indicator which equals 1 if $Y_n = k$ and 0 otherwise. Then the joint likelihood and log-likelihood are given by


$$
\begin{align*}
L\left(\bpi; \{\bx_n, Y_n\}_{n = 1}^N\right) &= \prodN \prod_{k = 1}^K \pi_k^{I_{nk}} 
\\
\log L\left(\bpi; \{\bx_n, Y_n\}_{n = 1}^N\right) &= \sumN \sum_{k = 1}^K I_{nk} \log(\pi_k)
\\
&= \sum_{k = 1}^K N_k\log(\pi_k),
\end{align*}
$$


where $N_k = \sumN I_{nk}$ gives the number of observations in class $k$ for $k = 1, \dots, K$. 



```{admonition} Math Note
The *Lagrangian function* provides a method for optimizing a function $f(\bx)$ subject to the constraint $g(\bx) = 0$.  The Lagrangian is given by 

$$
\mathcal{L}(\lambda, \bx) = f(\bx) - \lambda g(\bx).
$$

$\lambda$ is known as the *Lagrange multiplier*. The critical points of $f(\bx)$ (subject to the equality constraint) are found by setting the gradients of $\mathcal{L}(\lambda, \bx)$ with respect to $\lambda$ and $\bx$ equal to 0. 
```



Noting the constraint $\sum_{k = 1}^K \pi_k = 1$ (or equivalently $\sum_{k = 1}^K\pi_k - 1 = 0$), we can maximize the log-likelihood with the following Lagrangian. 


$$
\begin{align*}
\mathcal{L}(\bpi) &= \sum_{k = 1}^K N_k \log(\pi_k) - \lambda(\sum_{k = 1}^K \pi_k - 1).
\\
\dadb{\mathcal{L}(\bpi)}{\pi_k} &= \frac{N_k}{\pi_k} - \lambda, \hspace{3mm}\forall \hspace{1mm}k \in \{1, \dots, K\}
\\
\dadb{\mathcal{L}(\bpi)}{\lambda} &= 1 - \sum_{k = 1}^K \pi_k.
\\
\end{align*}
$$


This system of equations gives an intuitive solution:


$$
\hat{\pi}_k = \frac{N_k}{N}, \hspace{1mm} \lambda = N,
$$


which says that our estimate of $p(Y_n = k)$ is just the sample fraction of observations from class $k$. 



### 2.2 Data Likelihood

The next step is to model the conditional distribution of $\bx_n$ given $Y_n$ so that we can estimate this distribution's parameters. This of course depends on the family of distributions we choose to model $\bx_n$. Three common approaches are detailed below. 



#### 2.2.1 Linear Discriminative Analysis (LDA)

In LDA, we assume 


$$
\bx_n|(Y_n = k) \sim \text{MVN}(\bmu_k, \bSigma),
$$


for $k = 1, \dots, K$. Note that each class has the same covariance matrix but a unique mean vector.

Let's derive the parameters in this case. First, let's find the likelihood and log-likelihood. Note that we can write the joint likelihood as follows,


$$
L\l\{\bmu_k\}_{k = 1}^K, \bSigma\r = \prodN \prodK \Big(p\l\bx_n|\bmu_k, \bSigma\r\Big)^{I_{nk}},
$$


since $\left(p(\bx_{n}|\bmu_k, \bSigma)\right)^{I_{nk}}$ equals 1 if $y_n \neq k$ and $p(\bx_n|\bmu_k, \bSigma)$ otherwise. Then we plug in the Multivariate Normal PDF (dropping multiplicative constants) and take the log, as follows. 


$$
\begin{align*}
L\l\{\bmu_k\}_{k = 1}^K, \bSigma\r &= \prodN\prodK \Big(\frac{1}{\sqrt{|\bSigma|}}\exp\left\{-\frac{1}{2}(\bx_n - \bmu_k)^\top\bSigma^{-1}(\bx_n - \bmu_k)\right\}\Big)^{I_{nk}} 
\\
\log L\l\{\bmu_k\}_{k = 1}^K, \bSigma\r &= \sumN\sumK I_{nk}\l-\frac{1}{2} \log|\bSigma| -\frac{1}{2}(\bx_n - \bmu_k)^\top\bSigma^{-1}(\bx_n - \bmu_k) \r
\end{align*}
$$


```{admonition} Math Note
The following matrix derivatives will be of use for maximizing the above log-likelihood. 

For any invertible matrix $\mathbf{W}$,

$$
\dadb{|\mathbf{W}|}{\mathbf{W}} = |\mathbf{W}|\mathbf{W}^{-\top}, \tag{1}
$$

where $\mathbf{W}^{-\top} = (\mathbf{W}^{-1})^\top$. It follows that 

$$
\dadb{\log |\mathbf{W}|}{\mathbf{W}} = \mathbf{W}^{-\top}. \tag{2}
$$

We also have 

$$
\dadb{\bx^\top \mathbf{W}^{-1} \bx}{\mathbf{W}} = -\mathbf{W}^{-\top} \bx \bx^\top \mathbf{W}^{-\top}. \tag{3}
$$


For any symmetric matrix $\mathbf{A}$, 

$$
\dadb{(\bx - \mathbf{s})^\top \mathbf{A} (\bx - \mathbf{s})}{\mathbf{s}} = -2\mathbf{A}(\bx - \mathbf{s}). \tag{4}
$$

These results come from the [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf).
```



Let's start by estimating $\bSigma$. First, simplify the log-likelihood to make the gradient with respect to $\bSigma$ more apparent.


$$
\log L\l\{\bmu_k\}_{k = 1}^K, \bSigma\r = - \frac{N}{2}\log |\bSigma| -\frac{1}{2}  \sumN\sumK I_{nk}(\bx_n - \bmu_k)^\top\bSigma^{-1}(\bx_n - \bmu_k).
$$


Then, using equations (2) and (3) from the *Math Note*, we get


$$
\begin{align*}
\dadb{\log L\l\{\bmu_k\}_{k = 1}^K, \bSigma\r }{\bSigma} &= -\frac{N}{2}\bSigma^{-\top} + \frac{1}{2}\sumN\sumK I_{nk}\bSigma^{-\top} (\bx_n - \bmu_k)(\bx_n - \bmu_k)^\top\bSigma^{-\top}.
\end{align*}
$$


Finally, we set this equal to 0 and multiply by $\bSigma^{-1}$ on the left to solve for $\hat{\bSigma}$:


$$
\begin{align*}
0 &= -\frac{N}{2} + \frac{1}{2}\l\sumN\sumK I_{nk} (\bx_n - \bmu_k)(\bx_n - \bmu_k)^\top\r\bSigma^{-\top} \\
\bSigma^\top &= \frac{1}{N}\sumN \sumK I_{nk}(\bx_n - \bmu_k)(\bx_n - \bmu_k)^\top \\
\hat{\bSigma} &= \frac{1}{N}\mathbf{S}_T,
\end{align*}
$$


where $\mathbf{S}_T = \sumN\sumK I_{nk}(\bx_n - \bmu_k)(\bx_n - \bmu_k)^\top$. 



Now, to estimate the $\bmu_k$, let's look at each class individually. Let $N_k$ be the number of observations in class $k$ and $C_k$ be the set of observations in class $k$. Looking only at terms involving $\bmu_k$, we get


$$
\begin{align*}
\log L(\bmu_k, \bSigma) &= -\frac{1}{2} \sum_{n \in C_k} \Big( \log|\bSigma| + (\bx_n - \bmu_k)^\top \bSigma^{-1}(\bx_n - \bmu_k) \Big).
\end{align*}
$$


Using equation (4) from the *Math Note*, we calculate the gradient as 


$$
\dadb{\log L(\bmu_k, \bSigma)}{\bmu_k} =\sum_{n \in C_k} \bSigma^{-1}(\bx_n - \bmu_k).
$$


Setting this gradient equal to 0 and solving, we obtain our $\bmu_k$ estimate:


$$
\hat{\bmu}_k = \frac{1}{N_k} \sum_{n \in C_k} \bx_n = \bar{\bx}_k,
$$


where $\bar{\bx}_k$ is the element-wise sample mean of all $\bx_n$ in class $k$.



#### 2.2.2 Quadratic Discriminative Analysis (QDA)

QDA looks very similar to LDA but assumes each class has its *own* covariance matrix:


$$
\bx_n|(Y_n = k) \sim \text{MVN}(\bmu_k, \bSigma_k)
$$


for $k = 1, \dots, K$. The log-likelihood is the same as in LDA except we replace $\bSigma$ with $\bSigma_k$:


$$
\begin{align*}
\log L\l\{\bmu_k, \bSigma_k\}_{k = 1}^K\r &= \sumN\sumK I_{nk}\l-\frac{1}{2} \log|\bSigma_k| -\frac{1}{2}(\bx_n - \bmu_k)^\top\bSigma_k^{-1}(\bx_n - \bmu_k) \r.
\end{align*}
$$


Again, let's look at the parameters for each class individually. The log-likelihood for class $k$ is given by


$$
\begin{align*}
\log L(\bmu_k, \bSigma_k) &= -\frac{1}{2} \sum_{n \in C_k} \Big( \log|\bSigma_k| + (\bx_n - \bmu_k)^\top \bSigma_k^{-1}(\bx_n - \bmu_k) \Big). 
\end{align*}
$$


We could take the gradient of this log-likelihood with respect to $\bmu_k$ and set it equal to 0 to solve for $\hat{\bmu}_k$. However, we can also note that our $\hat{\bmu}_k$ estimate from the LDA approach will hold since this expression didn't depend on the covariance term (which is the only thing we've changed). Therefore, we again get


$$
\hat{\bmu}_k = \bar{\bx}_k.
$$


To estimate the $\bSigma_k$, we take the gradient of the log-likelihood for class $k$.


$$
\begin{align*}
\dadb{\log L(\bmu_k, \bSigma_k) }{\bSigma_k} &= -\frac{1}{2}\sum_{n \in C_k} \left( \bSigma_k^{-\top} - \bSigma_k^{-\top}(\bx_n - \bmu_k)(\bx_n - \bmu_k)^\top \bSigma_k^{-\top} \right).
\end{align*}
$$


Then we set this equal to 0 to solve for $\hat{\bSigma}_k$:


$$
\begin{align*}
\sum_{n \in C_k} \bSigma_k^{-\top}  &= \sum_{n \in C_k} \bSigma_k^{-\top}(\bx_n - \bmu_k)(\bx_n - \bmu_k)^\top \bSigma_k^{-\top}  \\
N_k I &= \sum_{n \in C_k} (\bx_n - \bmu_k)(\bx_n - \bmu_k)^\top \bSigma_k^{-\top} \\ 
\bSigma_k^\top  &= \frac{1}{N_k} \sum_{n \in C_k} (\bx_n - \bmu_k)(\bx_n - \bmu_k)^\top \\
\hat{\bSigma}_k &= \frac{1}{N_k} \mathbf{S}_k,
\end{align*}
$$


where $\mathbf{S}_k = \sum_{n \in C_k} (\bx_n - \bmu_k)(\bx_n - \bmu_k)^\top$. 



#### 2.2.3 Naive Bayes

Naive Bayes assumes the random variables within $\bx_n$ are independent conditional on the class of observation $n$. I.e. if $\bx_n \in \mathbb{R}^D$, Naive Bayes assumes 


$$
p(\bx_n|Y_n) = p(x_{n1}|Y_n)\cdot p(x_{n2}|Y_n) \cdot ... \cdot p(x_{nD}|Y_n).
$$


This makes estimating $p(\bx_n|Y_n)$ very easy—to estimate the parameters of $p(x_{nd}|Y_n)$, we can ignore all the variables in $\bx_{n}$ other than $x_{nd}$!

As an example, assume $\bx_n \in \mathbb{R}^2$ and we use the following model (where for simplicity $n$ and $\sigma^2_k$ are known). 


$$
\begin{align*}
x_{n1}|(Y_n = k) &\sim \mathcal{N}(\mu_k, \sigma^2_k) \\
x_{n2}|(Y_n = k) &\sim \text{Bin}(n, p_k).
\end{align*}
$$


Let the $\btheta_k = (\mu_k, \sigma_k^2, p_k)$ contain all the parameters for class $k$ . The joint likelihood function would become


$$
\begin{align*}
L(\{\btheta_k\}_{k = 1}^K) &= \prodN \prodK \l p(\bx|Y_n, \btheta_k)\r^{I_{nk}} \\
L(\{\btheta_k\}_{k = 1}^K) &= \prodN \prodK \l p(x_{n1}|\mu_k, \sigma^2_k) \cdot p(x_{n2}|p_k) \r^{I_{nk}},
\end{align*}
$$


where the two are equal because of the Naive Bayes conditional independence assumption. This allows us to easily find maximum likelihood estimates. The rest of this sub-section demonstrates how those estimates would be found, though it is nothing beyond ordinary maximum likelihood estimation. 



The log-likelihood is given by 


$$
\log L(\{\btheta_k\}_{k = 1}^K) = \sumN\sumK I_{nk}\l\log p(x_{n1}|\mu_k, \sigma^2_k) + \log p(x_{n2}|p_k) \r.
$$


As before, we estimate the parameters in each class by looking only at the terms in that class. Let's look at the log-likelihood for class $k$: 


$$
\begin{align*}
\log L(\btheta_k) &= \sum_{n \in C_k} \log p(x_{n1}|\mu_k, \sigma^2_k) + \log p(x_{n2}|p_k) \\ 
&= \sum_{n \in C_k} -\frac{(x_{n1} - \mu_k)^2}{2\sigma^2_k} + x_{n2}\log(p_k) + (1-x_{n2})\log(1-p_k).
\end{align*}
$$


Taking the derivative with respect to $p_k$, we're left with 


$$
\dadb{\log L(\btheta_k)}{p_k} = \sum_{n \in C_k}\frac{x_{n2}}{p_k} - \frac{1-x_{n2}}{1-p_k},
$$


which, will give us $\hat{p}_k = \frac{1}{N_k}\sum_{n \in C_k} x_{n2}$ as usual. The same process would again give typical results for $\mu_k$ and and $\sigma^2_k$. 



## 3. Making Classifications



Regardless of our modeling choices for $p(\bx_n|Y_n)$, classifying new observations is easy. Consider a test observation $\bx_0$. For $k = 1, \dots, K$, we use Bayes' rule to calculate 


$$
\begin{align*}
p(Y_0 = k|\bx_0) &\propto p(\bx_0|Y_0 = k)p(Y_0 = k) 
\\
&= \hat{p}(\bx_0|Y_0 = k)\hat{\pi}_k,
\end{align*}
$$


where $\hat{p}$ gives the estimated density of $\bx_0$ conditional on $Y_0$. We then predict $Y_0 = k$ for whichever value $k$ maximizes the above expression.

