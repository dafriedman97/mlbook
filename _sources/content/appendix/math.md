# Math

$$
\newcommand{\sumN}{\sum_{n = 1}^N}
\newcommand{\sumn}{\sum_n}
\newcommand{\prodN}{\prod_{n = 1}^N}
\newcommand{\by}{\mathbf{y}} \newcommand{\bX}{\mathbf{X}}
\newcommand{\bx}{\mathbf{x}}
\newcommand{\bu}{\mathbf{u}}
\newcommand{\bv}{\mathbf{v}}
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

For a book on mathematical derivations, this text assumes knowledge of relatively few mathematical methods. Most of the mathematical background required is summarized in the three following sections on calculus, matrices, and matrix calculus. 



## Calculus

The most important mathematical prerequisite for this book is calculus. Almost all of the methods covered involve minimizing a loss function or maximizing a likelihood function, done by taking the function's derivative with respect to one or more parameters and setting it equal to 0. 

Let's start by reviewing some of the most common derivatives used in this book:


$$
\begin{align*}
f(x) &= x^a \rightarrow f'(x)  = ax^{a-1} \\
f(x) &= \exp(x) \rightarrow  f'(x)  = \exp(x) \\ 
f(x) &= \log(x) \rightarrow f'(x) = \frac{1}{x} \\
f(x) &= |x| \rightarrow f'(x) = \begin{cases} 1, & x > 0 \\ -1, &  x < 0,\end{cases} \\
\end{align*}
$$


We will also often use the sum, product, and quotient rules:


$$
\begin{align*}
f(x) &= g(x) + h(x) \rightarrow f'(x) = g'(x) + h'(x) \\
f(x) &= g(x)\cdot h(x) \rightarrow f'(x)= g'(x)h(x) + g(x)h'(x)\\
f(x) &= g(x)/h(x) \rightarrow f'(x) = \frac{h(x)g'(x) + g(x)h'(x)}{h(x)^2}.
\end{align*}
$$
Finally, we will heavily rely on the chain rule:


$$
f(x) = g(h(x)) \rightarrow f'(x) = g'(h(x))h'(x).
$$



As an example of the chain rule, suppose $f(x) = \log(x^2)$. Let $h(x) = x^2$, meaning $f(x) = \log(h(x))$. Then


$$
f'(x) = \frac{1}{h(x)}h'(x) = \frac{1}{x^2}\cdot2x = \frac{2}{x}. 
$$



## Matrices 

While little linear algebra is used in this book, matrix and vector representations of data are very common. The most important matrix and vector operations are reviewed below. 

Let $\mathbf{u}$ and $\mathbf{v}$ be two column vectors of length $D$. The **dot product** of $\mathbf{u}$ and $\mathbf{v}$ is a scalar value given by 

$$
\mathbf{u} \cdot \mathbf{v} = \mathbf{u}^\top \mathbf{v} = \sum_{d = 1}^D u_d v_d = u_1v_1 + u_2v_2 + \dots + u_Dv_D.
$$
If $\bv$ is a vector of features (with a leading 1 appended for the intercept term) and $\bu$ is a vector of weights, this dot product is also referred to as a *linear combination* of the predictors in $\bv$. 

The **L1 norm** and **L2 norm** measure a vector's magnitude. For a vector $\bu$, these are given respectively by


$$
\begin{align}
||\bu||_1 &= \sum_{d = 1}^D |u_d| \\
||\bu||_2 &= \sqrt{\sum_{d = 1}^D u_d^2}. \\
\end{align}
$$


Let $\mathbf{A}$ be a $(N \times D)$ matrix defined as 

$$
\mathbf{A} = \begin{pmatrix} A_{11} & A_{12} & ... & A_{1D}  \\ 
A_{21} & A_{22} & ... & A_{2D} \\
& & ... & \\
A_{N1} & A_{N2} & ... & A_{ND} \end{pmatrix}
$$


The transpose of $\mathbf{A}$ is a $(D \times N)$ matrix given by 


$$
\mathbf{A}^T = \begin{pmatrix} A_{11} & A_{21} & ... & A_{N1} \\
A_{12} & A_{22} & ... & A_{N2} \\
& & ... & \\
A_{1D} & A_{2D} & ... & A_{ND} \end{pmatrix}
$$


If $\mathbf{A}$ is a square $(N \times N)$ matrix, its inverse, given by $\mathbf{A}^{-1}$, is the matrix such that 


$$
\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = I_N.
$$


## Matrix Calculus

Dealing with multiple parameters, multiple observations, and sometimes multiple loss functions, we will often have to take multiple derivatives at once in this book. This is done with matrix calculus. 

In this book, we will use the [numerator layout](https://en.wikipedia.org/wiki/Matrix_calculus#Numerator-layout_notation) convention for matrix derivatives. This is most easily shown with examples. First, let $a$ be a scalar and $\mathbf{u}$ be a vector of length $I$. The derivative of $a$ with respect to $\bu$ is given by


$$
\dadb{a}{\mathbf{u}} = \begin{pmatrix} \dadb{a}{u_1} & .. & \dadb{a}{u_I}\end{pmatrix} \in \R^{I},
$$


and the derivative of $\bu$ with respect to $a$ is given by 


$$
\dadb{\bu}{a} = \begin{pmatrix} \dadb{u_1}{a} \\ ... \\ \dadb{u_I}{a} \end{pmatrix} \R^I.
$$


Note that in either case, the first dimension of the derivative is determined by what's in the numerator. Similarly, letting $\bv$ be a vector of length $J$, the derivative of $\bu$ with respect to $\bv$ is given with 


$$
\dadb{\bu}{\bv} = \begin{pmatrix} \dadb{u_1}{v_1} & ... & \dadb{u_1}{v_J} \\ & ... & \\ \dadb{u_I}{v_1} & ... & \dadb{u_I}{v_J}\end{pmatrix} \in \R^{I \times J}.
$$


We will also have to take derivatives of or with respect to matrices. Let $\bX$ be a $(N \times D)$ matrix. The derivative of $\bX$ with respect to a constant $a$ is given by


$$
\dadb{\bX}{a} = \begin{pmatrix} \dadb{X_{11}}{a} & ... & \dadb{X_{1D}}{a} \\ & ... & \\ \dadb{X_{N1}}{a} & ... & \dadb{X_{ND}}{a}\end{pmatrix}  \in \R^{N \times D},
$$


and conversely the derivative of $a$ with respect to $\bX$ is given by 


$$
\dadb{a}{\bX} =  \begin{pmatrix} \dadb{a}{X_{11}} & ... & \dadb{a}{X_{1D}} \\ & ... & \\ \dadb{a}{X_{N1}} & ... & \dadb{a}{X_{ND}} \end{pmatrix}  \in \R^{N \times D}.
$$


Finally, we will occasionally need to take derivatives of vectors with respect to matrices or vice versa. This results in a *tensor* of 3 or more dimensions. Two examples are given below. First, the derivative of $\bu \in \R^I$ with respect to $\bX \in \R^{N \times D}$ is given by


$$
\dadb{\bu}{\bX} = \begin{pmatrix} \begin{pmatrix} \dadb{u_1}{X_{11}} & ... & \dadb{u_1}{X_{1D}} \\ & ... & \\ \dadb{u_1}{X_{N1}} & ... & \dadb{u_1}{X_{ND}} \end{pmatrix}
& ... &
\begin{pmatrix} \dadb{u_I}{X_{11}} & ... & \dadb{u_I}{X_{1D}} \\ & ... & \\ \dadb{u_I}{X_{N1}} & ... & \dadb{u_I}{X_{ND}}  \end{pmatrix} \end{pmatrix}  \in \R^{I \times N \times D},
$$


and the derivative of $\bX$ with respect to $\bu$ is given by


$$
\dadb{\bX}{\bu} = \begin{pmatrix} \begin{pmatrix} \dadb{X_{11}}{u_1} & ... & \dadb{X_{11}}{u_I} \end{pmatrix}
& ... &
\begin{pmatrix}  \dadb{X_{1D}}{u_1} & ... & \dadb{X_{1D}}{u_I}\end{pmatrix} \\
& ... & \\
 \begin{pmatrix}\dadb{X_{N1}}{u_1} & ... & \dadb{X_{N1}}{u_I} \end{pmatrix}
& ... &
\begin{pmatrix}\dadb{X_{ND}}{u_1} & ... & \dadb{X_{ND}}{u_I}\end{pmatrix} \\
\end{pmatrix}  \in \R^{N \times D \times I}.
$$




Notice again that what we are taking the derivative *of* determines the first dimension(s) of the derivative and what we are taking the derivative with respect *to* determines the last. 