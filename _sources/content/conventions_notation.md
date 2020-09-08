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

# Conventions and Notation



The following terminology will be used throughout the book.  

- Variables can be split into two types: the variables we intend to model are referred to as **target** or **output** variables, while the variables we use to model the target variables are referred to as **predictors**, **features**, or **input** variables. These are also known as the *dependent* and *independent* variables, respectively.
- An **observation** is a single collection of predictors and target variables. Multiple observations with the same variables are combined to form a **dataset**. 
- A **training** dataset is one used to build a machine learning model. A **validation** dataset is one used to compare multiple models built on the same training dataset with different parameters. A **testing** dataset is one used to evaluate a final model. 
- Variables, whether predictors or targets, may be **quantitative** or **categorical**. Quantitative variables follow a continuous or near-contih234nuous scale (such as height in inches or income in dollars). Categorical variables fall in one of a discrete set of groups (such as nation of birth or species type). While the values of categorical variables may follow some natural order (such as shirt size), this is not assumed. 
- Modeling tasks are referred to as **regression** if the target is quantitative and **classification** if the target is categorical. Note that regression does not necessarily refer to ordinary least squares (OLS) linear regression. 



Unless indicated otherwise, the following conventions are used to represent data and datasets. 

- Training datasets are assumed to have $N$ observations and $D$ predictors.

- The vector of features for the $n^\text{th}$ observation is given by $\bx_n$. Note that $\bx_n$ might include functions of the original predictors through feature engineering. When the target variable is single-dimensional (i.e. there is only one target variable per observation), it is given by $y_n$; when there are multiple target variables per observation, the vector of targets is given by $\by_n$.

- The entire collection of input and output data is often represented with $\{\bx_n, y_n\}_{n = 1}^N$, which implies observation $n$ has a multi-dimensional predictor vector $\bx_n$ and a target variable $y_n$ for $n = 1, 2, \dots, N$. 

- Many models, such as ordinary linear regression, append an intercept term to the predictor vector. When this is the case, $\bx_n$ will be defined as 
  
  $$
  \bx_n = \begin{pmatrix} 1 & x_{n1} & x_{n2} & ... & x_{nD} \end{pmatrix}.
  $$

  

- *Feature matrices* or *data frames* are created by concatenating feature vectors across observations. Within a matrix, feature vectors are row vectors, with $\bx_n$ representing the matrix's $n^\text{th}$ row. These matrices are then given by $\bX$. If a leading 1 is appended to each $\bx_n$, the first column of the corresponding feature matrix $\bX$ will consist of only 1s. 

  

Finally, the following mathematical and notational conventions are used.

- Scalar values will be non-boldface and lowercase, random variables will be non-boldface and uppercase, vectors will be bold and lowercase, and matrices will be bold and uppercase. E.g. $b$ is a scalar, $B$ a random variable, $\mathbf{b}$ a vector, and $\mathbf{B}$ a matrix. 

- Unless indicated otherwise, all vectors are assumed to be column vectors. Since feature vectors (such as $\bx_n$ and $\bphi_n$ above) are entered into data frames as rows, they will sometimes be treated as row vectors, even outside of data frames.

- Matrix or vector derivatives, covered in the {doc}`math appendix </content/appendix/math>`, will use the numerator [layout convention](https://en.wikipedia.org/wiki/Matrix_calculus#Layout_conventions). Let $\by \in \R^a$ and $\bx \in \R^b$; under this convention, the derivative $\partial\by/\partial\bx$ is written as 
  
$$
  \dadb{\by}{\bx} = \begin{pmatrix}
  \dadb{y_1}{x_1} & ... & \dadb{y_1}{x_b} \\
  \dadb{y_2}{x_1} &  ... & \dadb{y_2}{x_b} \\
  &  ... & \\
  \dadb{y_a}{x_1} & ... & \dadb{y_a}{x_b} \\
  \end{pmatrix}.
  $$



- The likelihood of a parameter $\theta$ given data $\{x_n\}_{n = 1}^N$ is represented by $\mathcal{L}\left(\theta; \{x_n\}_{n = 1}^N\right)$. If we are considering the data to be random (i.e. not yet observed), it will be written as $\{X_n\}_{n = 1}^N$. If the data in consideration is obvious, we may write the likelihood as just $\mathcal{L}(\theta)$. 

