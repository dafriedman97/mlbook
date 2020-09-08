# The Perceptron Algorithm

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

The *perceptron algorithm* is a simple classification method that plays an important historical role in the development of the much more flexible {doc}`neural network </content/c7/concept>`. The perceptron is a *linear binary classifier*—*linear* since it separates the input variable space linearly and *binary* since it places observations into one of two classes. 



## Model Structure

It is most convenient to represent our binary target variable as $y_n \in \{-1, + 1\}$. For example, an email might be marked as $+1$ if it is spam and $-1$ otherwise. As usual, suppose we have one or more predictors per observation. We obtain our feature vector $\bx_n$ by concatenating a leading 1 to this collection of predictors.

Consider the following function, which is an example of an *activation function*:


$$
f(x) = 
\begin{cases}
+1, &x \geq 0\\
-1, &x <0.
\end{cases}
$$


The perceptron applies this activation function to a linear combination of $\mathbf{x}_n$ in order to return a fitted value. That is,


$$
\hat{y}_n = f(\bbetahat^\top\bx_n).
$$


In words, the perceptron predicts $+1$ if $\bbetahat^\top\bx_n \geq 0$ and $-1$ otherwise. Simple enough!



Note that an observation is correctly classified if $y_n\hat{y}_n = 1$ and misclassified if $y_n \hat{y}_n = -1$. Then let $\mathcal{M}$ be the set of misclassified observations, i.e. all $n \in \{1, \dots, N\}$ for which $y_n\hat{y}_n = -1$.



## Parameter Estimation

As usual, we calculate the $\bbetahat$ as the set of coefficients to minimize some loss function. Specifically, the perceptron attempts to minimize the *perceptron criterion*, defined as


$$
\mathcal{L}(\bbetahat) = -\sum_{n \in \mathcal{M}} y_n(\bbetahat^\top \bx_n).
$$


The perceptron criterion does not penalize correctly classified observations but penalizes misclassified observations based on the magnitude of $\bbetahat^\top \bx_n$—that is, how wrong they were. 

The gradient of the perceptron criterion is 


$$
\dadb{\mathcal{L}(\bbetahat)}{\bbetahat} = - \sum_{n \in \mathcal{M}} y_n \bx_n. 
$$


We obviously can't set this equal to 0 and solve for $\bbetahat$, so we have to estimate $\bbetahat$ through {doc}`gradient descent </content/appendix/methods>`. Specifically, we could use the following procedure, where $\eta$ is the learning rate. 



1. Randomly instantiate $\bbetahat$
2. Until convergence or some stopping rule is reached:
   1. For $n \in \{1, \dots, N\}$:
      1.  $\hat{y}_n = f(\bbetahat^\top \bx_n)$ 
      2. If $y_n\hat{y}_n  = -1$, $\hspace{5mm} \bbetahat \leftarrow \bbetahat + \eta\cdot y_n\bx_n$.



It can be shown that convergence is guaranteed in the linearly separable case but not otherwise. If the classes are not linearly separable, some stopping rule will have to be determined. 