# Bagging



Bagging, short for *bootstrap aggregating*, combines the results of several learners trained on bootstrapped samples of the training data. The process of bagging is very simple yet often quite powerful.  



## Bootstrapping 

Bootstrapping is commonly used in statistical problems to approximate the variance of an estimator. Suppose we observe dataset $\mathcal{D}$ and we are interested in some parameter $\theta$ (such as the mean or median). Estimating $\theta$ is one thing, but understanding the *variance* of that estimate is another. For difficult problems, the bootstrap helps find this variance. 

The variance of an estimator tells us how much it would vary from sample to sample. Suppose we drew $B$ datasets of equal size, $\mathcal{D}_1$ through $\mathcal{D}_B$, and each time estimated of $\theta$. We would then be able to calculate the variance of our estimate since we have $B$ observed estimates. In practice however, we are stuck with only one dataset. The bootstrap enables us to approximate the variance of our estimate with only one dataset. This procedure is outlined below. 



____

**Bootstrap Procedure**

Suppose we have a dataset $\mathcal{D} = \{x_1, x_2, \dots, x_N\}$ of size $N$. Suppose also that we choose to estimate the parameter $\theta$ with some function of the data, $\hat{\theta} = f(\mathcal{D})$. For some large value $B$, do the following. 

1. For $b = 1, 2, \dots, B$,
   1. Sample $N$ observations from $\mathcal{D}$ *with* replacement. Call this sample $\mathcal{D}^*_b$. 
   2. Estimate $\theta$ on the bootstrapped sample, $\hat{\theta}^*_b = f(\mathcal{D}^*_b)$. 
2. Calculate the sample variance of the bootstrapped estimates, $\{ \hat{\theta}^*_1, \dots, \hat{\theta}^*_B\}$. Let this be our estimate of the variance of $\hat{\theta}$. 

_____



The same bootstrap procedure can be extended to serve a very different purpose. Bagging uses the bootstrap to *reduce*—rather than *estimate*—the variance of a model. 



## Bagging for Decision Trees



A bagging model trains many learners on bootstrapped samples of the training data and aggregates the results into one final model. We will outline this procedure for decision trees, though the approach works for other learners. 



____

**Bagging Procedure**

Given a training dataset $\mathcal{D} = \{\bx_n, y_n\}_{n = 1}^N$ and a separate test set $\mathcal{T} = \{\bx_t \}_{t = 1}^T$, we build and deploy a bagging model with the following procedure. The first step builds the model (the learners) and the second generates fitted values. 

1. For $b =  1, 2, \dots, B$,
   1. Draw a bootstrapped sample $\mathcal{D}^*_b$. 
   2. Build a decision tree $T^*_b$ to the bootstrapped sample.
2. For test observation $t = 1, 2, \dots, T$,
   1. For $b = 1, 2, \dots, B$,
      1. Calculate $\hat{y}^*_{tb}$, observation $t$'s fitted value according to tree $T^*_b$.
   2. Combine the $\hat{y}^*_{tb}$  into a single fitted value, $\hat{y}_t$.

____



How exactly we combine the results of the learners into a single fitted value (the second part of the second step) depends on the target variable. For a continuous target variable, we typically average the learners' predictions. For a categorical target variable, we typically use the class that receives the plurality vote. 



To recap bagging, we simply build a large number of learners on bootstrapped samples of the data and combine the predictions of these learners into one single fitted value. As if creating new datasets, the bootstrap provides our model with more training samples in order to improve its estimates. 