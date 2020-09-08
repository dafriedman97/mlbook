# Random Forests



A random forest is a slight extension to the bagging approach for decision trees that can further decrease overfitting and improve out-of-sample precision. Unlike bagging, random forests are exclusively designed for decision trees (hence the name). 

Like bagging, a random forest combines the predictions of several base learners, each trained on a bootstrapped sample of the original training set. Random forests, however, add one additional regulatory step: at each split within each tree, we only consider splitting a randomly-chosen subset of the predictors. In other words, we explicitly prohibit the trees from considering some of the predictors in each split. 

It might seem counter-intuitive to restrict the amount of information available to our model while training. Recall that the advantage of bootstrapping is to provide the model with a greater sense of the training data by "generating" different datasets. If the base learners are too similar, however, we don't gain anything by averaging them. Randomly choosing eligible predictors at each split *de-correlates* the trees, serving to further differentiate the trees from one another. We then gain more information by averaging over trees which can result in significant improvements in precision.

The specific steps to building a random forest are provided below. 



____

**Random Forest Procedure**

Suppose we have dataset $\mathcal{D} = \{\bx_n, y_n\}_{n = 1}^N$ and test set $\mathcal{T} = \{\bx_t \}_{t = 1}^T$ where the predictors are $D$-dimensional (i.e. $\bx \in \R^D$). 

1. For $b =  1, 2, \dots, B$,
   1. Draw a bootstrapped sample $\mathcal{D}^*_b$. 
   2. Build a decision tree $T^*_b$ to the bootstrapped sample. At each split, only consider splitting along $C$ of the $D$ predictors (with $C \leq D$).
2. For test observation $t = 1, 2, \dots, T$,
   1. For $b = 1, 2, \dots, B$,
      1. Calculate $\hat{y}^*_{tb}$, observation $t$'s fitted value according to tree $T^*_b$.
   2. Combine the $\hat{y}^*_{tb}$  into a single fitted value, $\hat{y}_t$.

____



The only difference between the random forest procedure above and the bagging procedure in the previous section is the restricted use of predictors in the second part of the first step. If $C = D$, there is no difference between these two approaches. 



To recap, random forests average the results of several decision trees and add two sources of randomness to ensure differentiation between the base learners: randomness in which observations are sampled via the bootstrapping and randomness in which predictors are considered at each split.  

