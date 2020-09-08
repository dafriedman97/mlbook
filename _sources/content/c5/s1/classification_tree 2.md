# Classification Trees

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

Building a classification tree is essentially identical to building a regression tree but optimizing a different loss function—one fitting for a categorical target variable. For that reason, this section only covers the details unique to classification trees, rather than demonstrating how one is built from scratch. To understand the tree-building process in general, see the {doc}` previous section <regression_tree>`.

Suppose for the following that we have data $\{\bx_n, y_n\}_{n = 1}^N$ with predictor variables $\bx_n \in \R^D$ and a categorical target variable $y_n \in \{1, \dots, K\}$ . 



## Building a Tree



### The Objective

Two common loss functions for a classification are the *Gini index* and the *cross-entropy*. Let $n \in \mathcal{N}_m$ be the collection of training observations that pass through node $m$ and let $\hat{y}_{mk}$ be the fraction of these observations in class $k$ for $k = 1, \dots, K$. The Gini index for $\mathcal{N}_m$ is defined as


$$
\mathcal{L}_{G}(\mathcal{N}_m) = \sum_{k = 1}^K \hat{p}_{mk}(1-\hat{p}_{mk}),
$$


and the cross-entropy is defined as 


$$
\mathcal{L}_{E}(\mathcal{N}_m) = -\sum_{k = 1}^K \hat{p}_{mk} \log\hat{p}_{mk}.
$$

The Gini index and cross-entropy are measures of *impurity*—they are higher for nodes with more equal representation of different classes and lower for nodes represented largely by a single class. As a node becomes more pure, these loss measures tend toward zero.  

In order to evaluate the purity of a *split* (rather than that of a *node*), we use the weighted Gini index or weighted cross-entropy. Consider a split $S_m$ of bud $\mathcal{N}_m$ which creates children $\mathcal{C}_m^L$ and $\mathcal{C}_m^R$. Let the fraction of training observations going to $\mathcal{C}_m^L$ be $f_L$ and the fraction going to $\mathcal{C}_m^R$ be $f_R$. The weighted loss (whether with the Gini index or the cross-entropy) is defined as


$$
\mathcal{L}(S_m) = f_L\cdot \mathcal{L}(\mathcal{C}_m^L) +  f_R\cdot \mathcal{L}(\mathcal{C}_m^R).
$$


The lower the weighted loss the better. 



### Making Splits

As with regression trees, we will make splits one layer at a time. When splitting bud $m$, we use the same procedure as in regression trees: we calculate the loss from splitting the node at each value of each predictor and make the split with the lowest loss. 

For quantitative predictors, the procedure is identical to the regression tree procedure except we aim to minimize $\mathcal{L}(S_m)$ rather than maximally reducing $RSS_m$. For categorical predictors, we cannot rank the categories according to the average value of the target variable as we did for regression trees because the target is not continuous! If our target is binary, we can rank the predictor's categories according to the fraction of the corresponding target variables in class $1$ versus class $0$ and proceed in the same was as we did for regression trees. 



If the target is not binary, we are out of luck. One (potentially computationally-intensive) method is to try all possible binary groupings of the categorical value and see what grouping minimizes $\mathcal{L}(S_m)$. Another would be a one-versus-rest approach where we only consider isolating one category at a time from the rest. Suppose we had a predictor with four categories, $A, B, C,$ and $D$. The first method requires the following 7 splits while the second method requires only the first four splits. 


$$
\begin{align*}
A &\text{ vs. } B, C, D \\
B &\text{ vs. } A, C, D \\
C &\text{ vs. } A, B, D \\
D &\text{ vs. } A, C, D\\
A, B &\text{ vs. }C, D \\
A, C &\text{ vs. }B, D \\
A, D &\text{ vs. }B, C \\
\end{align*}
$$


### Making Predictions

Classifying test observations with a fully-grown tree is very straightforward. First, run an observation through the tree and observe which leaf it lands in. Then classify it according to the most common class in that leaf. 

For large enough leaves, we can also estimate the probability that the test observation belongs to any given class: if test observation $j$ lands in leaf $m$, we can estimate $p(y_j =  k)$ with $\hat{p}_{mk}$ for each $k$. 



## Choosing Hyperparameters 

In the regression tree section, we discussed three methods for managing a tree's size to balance the bias-variance tradeoff. The same three methods can be used for classification trees with slight modifications, which we cover next. For a full overview on these methods, please review the regression tree section. 



### Size Regulation

We can again use cross validation to fix the maximum depth of a tree or the minimum size of its terminal nodes. Unlike with regression trees, however, it is common to use a different loss function for cross validation than we do for building the tree. Specifically, we typically build classification trees with the Gini index or cross-entropy but use the *misclassification rate* to determine the hyperparameters with cross validation. The *misclassification rate* is simply the percent of observations we incorrectly classify. This is typically a more desirable metric to minimize than the Gini index or cross-entropy since it tells us more about our ultimate goal of correctly classifying test observations. 

To conduct cross validation, then, we would build the tree using the Gini index or cross-entropy for a set of hyperparameters, then pick the tree with the lowest misclassification rate on validation samples. 



### Maximum Split Loss

Another regularization method for regression trees was to require that each split reduce the $RSS$ by a certain amount. An equivalent approach for classification trees is to require that each split have a weighted loss below some minimum threshold. This threshold should be chosen through cross validation, again likely with the misclassification rate as the loss function. 



### Pruning

To prune a regression tree, we first fit a large tree, then searched for a sub-tree that could achieve a low $RSS$ without growing too large and possibly overfitting. Specifically, we looked for the sub-tree $T$ that minimized the following loss function, where $|T|$ gives the number of terminal leaves and $\lambda$ is a regularization parameter:


$$
\mathcal{L}(T) = RSS_T + \lambda|T|.
$$


We prune a classification tree in a nearly identical fashion. First, we grow an intentionally overfit tree, $T_0$. We then consider all splits leading to terminal nodes and undo the one with the greatest loss by re-joining its child nodes. I.e. we undo the split at the node $m$ with the greatest $\mathcal{L}(S_m)$ among nodes leading to leaves. We then repeat this process iteratively until we are left with the tree containing only the initial node. For each tree, we record its size (the number of terminal leaves) and its misclassification rate, $\mathcal{M}_T$. We then choose the tree that minimizes 


$$
\mathcal{L}(T) = \mathcal{M}_T + \lambda|T|, 
$$


where again $\lambda$ is chosen through cross validation. For a fuller overview of how we use cross validation to choose $\lambda$, see the pruning section in the {doc}`regression tree <regression_tree>` page. 