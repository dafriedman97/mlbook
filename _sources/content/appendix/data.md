# Datasets

The examples in this book use several datasets that are available either through `scikit-learn` or `seaboarn`. Those datasets are described briefly below. 



## Boston Housing 

The [Boston housing dataset](https://www.kaggle.com/c/boston-housing) contains information on 506 neighborhoods in Boston, Massachusetts. The target variable is the median value of owner-occupied homes (which appears to be censored at $50,000). This variable is approximately continuous, and so we will use this dataset for regression tasks. The predictors are all numeric and include details such as racial demographics and crime rates. It is available through `sklearn.datasets`.



## Breast Cancer

The [breast cancer dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) contains measurements of cells from 569 breast cancer patients. The target variable is whether the cancer is malignant or benign, so we will use it for binary classification tasks. The predictors are all quantitative and include information such as the perimeter or concavity of the measured cells. It is available through `sklearn.datasets`.



## Penguins

The [penguins dataset](https://www.kaggle.com/parulpandey/penguin-dataset-the-new-iris) contains measurements from 344 penguins of three different species: *Adelie*, *Gentoo*, and *Chinstrap*. The target variable is the penguin's species. The predictors are both quantitative and categorical, and include information from the penguin's flipper size to the island on which it was found. Since this dataset includes categorical predictors, we will use it for tree-based models (though one could use it for quantitative models by creating dummy variables). It is available through `seaborn.load_dataset()`



## Tips

The [tips dataset](https://www.kaggle.com/ranjeetjain3/seaborn-tips-dataset) contains 244 observations from a food server in 1990. The target variable is the amount of tips in dollars that the server received per meal. The predictors are both quantitative and categorical: the total bill, the size of the party, the day of the week, etc. Since the dataset includes categorical predictors and a quantitative target variable, we will use it for tree-based regression tasks. It is available through `seaborn.load_dataset()`. 



## Wine

The [wine dataset](https://archive.ics.uci.edu/ml/datasets/wine) contains data from chemical analysis on 178 wines of three classes. The target variable is the wine class, and so we will use it for classification tasks. The predictors are all numeric and detail each wine's chemical makeup. It is available through `sklearn.datasets`.





## 