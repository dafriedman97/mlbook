# Fisher's Linear Discriminant

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
\newcommand{\bmu}{\boldsymbol{\mu}}
\newcommand{\bSigma}{\boldsymbol{\Sigma}}
\newcommand{\bT}{\mathbf{T}}
\newcommand{\dadb}[2]{\frac{\partial #1}{\partial #2}}
\newcommand{\iid}{\overset{\small{\text{i.i.d.}}}{\sim}}
$$

Intuitively, a good classifier is one that bunches together observations in the same class and separates observations between classes. *Fisher's linear discriminant* attempts to do this through dimensionality reduction. Specifically, it projects data points onto a single dimension and classifies them according to their location along this dimension. As we will see, its goal is to find the projection that that maximizes the ratio of between-class variation to within-class variation. Fisher's linear discriminant can be applied to multiclass tasks, but we'll only review the binary case here. 



# Model Structure

As usual, suppose we have a vector of one or more predictors per observation, $\bx_n$. However we do *not* append a 1 to this vector. I.e., there is no bias term built into the vector of predictors. Then, we can project $\mathbf{x}_n$ to one dimension with 


$$
f(\mathbf{x}_n) = \bbeta^\top \bx_n.
$$


Once we've chosen our $\bbeta$, we can classify observation $n$ according to whether $f(\bx_n)$ is greater than some cutoff value. For instance, consider the data on the left below. Given the vector $\bbeta ^\top= \begin{bmatrix} 1 &-1 \end{bmatrix}$ (shown in red), we could classify observations as dark blue if $\bbeta^\top \bx_n \geq 2$ and light blue otherwise. The image on the right shows the projections using $\bbeta$. Using the cutoff $\bbeta^\top \bx_n \geq 2$, we see that most cases are correctly classified though some are misclassified. We can improve the model in two ways: either changing $\bbeta$ or changing the cutoff. 



![download-2](/content/c3/s1/img1.png)



In practice, the linear discriminant will tell us $\bbeta$ but won't tell us the cutoff value. Instead, the discriminant will rank the $f(\bx_n)$ so that the classes are separated as much as possible. It is up to us to choose the cutoff value. 



## Fisher Criterion 

The *Fisher criterion* quantifies how well a parameter vector $\bbeta$ classifies observations by rewarding between-class variation and penalizing within-class variation. The only variation it considers, however, is in the single dimension we project along. For each observation, we have 


$$
f(\mathbf{x}_n) = \bbeta^\top \bx_n.
$$


Let $N_k$ be the number of observations and $\mathcal{S}_k$ be the set of observations in class $k$ for $k \in \{0, 1\}$. Then let


$$
\bmu_k = \frac{1}{N_k}\sum_{n \in \mathcal{S}_k} \bx_n
$$


be the mean vector (also known as the *centroid*) of the predictors in class $k$. This class-mean is also projected along our single dimension with


$$
\mu_k = \bbeta^\top \bmu_k.
$$


A simple way to measure how well $\bbeta$ separates classes is with the magnitude of the difference between $\mu_2$ and $\mu_1$. To assess similarity *within* a class, we use 


$$
\sigma_k^2 = \sum_{n \in \mathcal{S}_k} \left(f(\bx_n) - \mu_k \right)^2,
$$


the within-class sum of squared differences between the projections of the observations and the projection of the class-mean. We are then ready to introduce the Fisher criterion: 


$$
F(\bbeta) = \frac{(\mu_2 - \mu_1)^2}{\sigma_1^2 + \sigma_2^2}.
$$


Intuitively, an increase in $F(\bbeta)$ implies the between-class variation has increased relative to the within-class variation. 



Let's write $F(\bbeta)$ as an explicit function of $\bbeta$. Starting with the numerator, we have 


$$
\begin{align*}
(\mu_2 - \mu_1)^2 &=  \left(\bbeta^\top(\bmu_2 - \bmu_1)\right)^2 
\\
&= \left(\bbeta^\top(\bmu_2 - \bmu_1)\right)\cdot \left(\bbeta^\top(\bmu_2 - \bmu_1)\right)
\\
&= \bbeta^\top(\bmu_2 - \bmu_1)(\bmu_2 - \bmu_1)^\top \bbeta 
\\
&= \bbeta^\top \bSigma_b\bbeta,
\end{align*}
$$


where $\bSigma_b = (\bmu_2 - \bmu_1)(\bmu_2 - \bmu_1)^\top$ is the *between class* covariance matrix. Then for the denominator, we have 


$$
\begin{align*}
\sigma_1^2 + \sigma_2^2 &= \sum_{n \in \mathcal{S}_1} \left(\bbeta^\top(\bx_n - \bmu_1) \right)^2 + \sum_{n \in \mathcal{S}_2} \left(\bbeta^\top(\bx_n - \bmu_2) \right)^2
\\
&= \sum_{n \in \mathcal{S}_1} \bbeta^\top(\bx_n - \bmu_1)(\bx_n - \bmu_1)^\top \bbeta + \sum_{n \in \mathcal{S}_2}  \bbeta^\top(\bx_n - \bmu_2)(\bx_n - \bmu_2)^\top \bbeta 
\\
&= \bbeta^\top \left(\sum_{n \in \mathcal{S}_1}(\bx_n - \bmu_1)(\bx_n - \bmu_1)^\top +  \sum_{n \in \mathcal{S}_2}(\bx_n - \bmu_2)(\bx_n - \bmu_2)^\top\right)\bbeta 
\\
&= \bbeta^\top \bSigma_w\bbeta,
\end{align*}
$$


where $\bSigma_w = \sum_{n \in \mathcal{S}_1}(\bx_n - \bmu_1)(\bx_n - \bmu_1)^\top +  \sum_{n \in \mathcal{S}_2}(\bx_n - \bmu_2)(\bx_n - \bmu_2)^\top$ is the *within class* covariance matrix.  In total, then, we have 


$$
F(\bbeta) = \frac{\bbeta^\top \bSigma_b\bbeta}{\bbeta^\top \bSigma_w \bbeta}.
$$


## Parameter Estimation

Finally, we can find the $\bbeta$ to optimize $F(\bbeta)$. Importantly, note that the magnitude of $\bbeta$ is unimportant since we simply want to rank the $f(\bx_n) = \bbeta^\top \bx_n$ values and using a vector proportional to $\bbeta$ will not change this ranking. 



```{admonition} Math Note
For a symmetric matrix $\mathbf{W}$ and a vector $\mathbf{s}$, we have 

$$
\dadb{\mathbf{s}^\top \mathbf{W}\mathbf{s}}{\mathbf{s}} = 2 \mathbf{W}\mathbf{s}.
$$

Notice that $\bSigma_w$ is symmetric since its $(i, j)^\text{th}$ element is 

$$
\sum_{n \in \mathcal{S}_1} (x_{ni} - \mu_{1i})(x_{nj} - \mu_{1j}) + \sum_{n \in \mathcal{S}_2} (x_{ni} - \mu_{2i})(x_{nj} - \mu_{2j}),
$$

which is equivalent to its $(j, i)^\text{th}$ element.
```



By the quotient rule and the math note above, 


$$
\begin{align*}
\dadb{F(\bbeta)}{\bbeta} &= \frac{2\bSigma_b\bbeta\left(\bbeta^\top \bSigma_w \bbeta\right) - 2\bSigma_w\bbeta\left(\bbeta^\top \bSigma_b \bbeta\right)}{(\bbeta^\top \bSigma_w\bbeta)^2}.
\end{align*}
$$


We then set this equal to 0. Note that the denominator is just a scalar, so it goes away. 


$$
\begin{align*}
\mathbf{0} &=\bSigma_b\bbeta\left(\bbeta^\top \bSigma_w \bbeta\right) - \bSigma_w\bbeta\left(\bbeta^\top \bSigma_b \bbeta\right) \\
\bSigma_w\bbeta\left(\bbeta^\top \bSigma_b \bbeta\right)  &= \bSigma_b\bbeta\left(\bbeta^\top \bSigma_w \bbeta\right).
\end{align*}
$$


Since we only care about the direction of $\bbeta$ and not its magnitude, we can make some simplifications. First, we can ignore $\bbeta^\top \bSigma_b \bbeta$ and $\bbeta^\top \bSigma_b \bbeta$ since they are just constants. Second, we can note that $\bSigma_b \bbeta$ is proportional to $\bmu_2 - \bmu_1$, as shown below:


$$
\bSigma_b \bbeta = (\bmu_2 - \bmu_1)(\bmu_2 - \bmu_1)^\top\bbeta = (\bmu_2 - \bmu_1)k\propto (\bmu_2 - \bmu_1),
$$


where $k$ is some constant. Therefore, our solution becomes


$$
\bbetahat \propto \bSigma_w^{-1}(\bmu_2 - \bmu_1).
$$


The image below on the left shows the $\bbetahat$ (in red) found by Fisher's linear discriminant. On the right, we again see the projections of these datapoints from $\bbetahat$. The cutoff is chosen to be around 0.05. Note that this discriminator, unlike the one above, successfully separates the two classes!



![](/content/c3/s1/img2.png)



