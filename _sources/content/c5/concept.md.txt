# Concept



A decision tree is an interpretable machine learning method for regression and classification. Trees iteratively split samples of the training data based on the value of a chosen predictor; the goal of each split is to create two sub-samples, or "children," with greater *purity* of the target variable than their "parent". For classification tasks, purity means the first child should have observations primarily of one class and the second should have observations primarily of another. For regression tasks, purity means the first child should have observations with high values of the target variable and the second should have observations with low values. 



An example of a classification decision tree using the {doc}`penguins dataset </content/appendix/data>` is given below. The tree attempts to classify a penguin's species—*Adelie*, *Gentoo*, or *Chinstrap*—from information about its flippers and bill. The first "node" shows that there are 333 training observations, 146 *Adelie*, 68 *Gentoo*, and 119 *Chinstrap*. We first split based on whether the penguin's flipper length is less than or equal to 206.5 mm. If so, the penguin moves to the node on the left and if not, it moves to the node on the right. We then repeat this process for each of the child nodes.



![tree](/content/c5/tree.png)



Once we've reached the bottom of the tree, we make our predictions. For instance, if a test observation has a `flipper_length` of 210 and a `bill_depth` of 12, we would follow the tree and classify it as a *Gentoo*. This simple decision process makes trees very interpretable.  However, they may suffer in terms of precision (accuracy of predictions) and robustness (sensitivity to variable training data). 



This chapter demonstrates how decision trees are built. The {doc}`first <s1/regression_tree>` section covers regression tasks, where the target variable is quantitative, and the {doc}`second <s1/classification_tree>` covers classification tasks, where the target variable is categorical. 