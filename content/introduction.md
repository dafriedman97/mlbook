# Introduction



![](/content/logo_light.png)



## What this Book Covers

This book covers the building blocks of the most common methods in machine learning. This set of methods is like a toolbox for machine learning engineers. Those entering the field of machine learning should feel comfortable with this toolbox so they have the right tool for a variety of tasks. Each chapter in this book corresponds to a single machine learning method or group of methods. In other words, each chapter focuses on a single tool within the ML toolbox.

In my experience, the best way to become comfortable with these methods is to see them derived from scratch, both in theory and in code. The purpose of this book is to provide those derivations. Each chapter is broken into three sections. The *concept* sections introduce the methods conceptually and derive their results mathematically. The *construction* sections show how to construct the methods from scratch using Python. The *implementation* sections demonstrate how to apply the methods using packages in Python like `scikit-learn`, `statsmodels`, and `tensorflow`. 



## Why this Book

There are many great books on machine learning written by more knowledgeable authors and covering a broader range of topics. In particular, I would suggest [An Introduction to Statistical Learning](http://faculty.marshall.usc.edu/gareth-james/ISL/), [Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/), and [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/), all of which are available online for free. 

While those books provide a conceptual overview of machine learning and the theory behind its methods, this book focuses on the bare bones of machine learning algorithms. Its main purpose is to provide readers with the ability to construct these algorithms independently. Continuing the toolbox analogy, this book is intended as a user guide: it is not designed to teach users broad practices of the field but rather how each tool works at a micro level.  



## Who this Book is for

This book is for readers looking to learn new machine learning algorithms or understand algorithms at a deeper level. Specifically, it is intended for readers interested in seeing machine learning algorithms derived from start to finish. Seeing these derivations might help a reader previously unfamiliar with common algorithms understand how they work intuitively. Or, seeing these derivations might help a reader experienced in modeling understand how different algorithms create the models they do and the advantages and disadvantages of each one.

This book will be most helpful for those with practice in basic modeling. It does not review best practices—such as feature engineering or balancing response variables—or discuss in depth when certain models are more appropriate than others. Instead, it focuses on the elements of those models. 



## What Readers Should Know

The *concept* sections of this book primarily require knowledge of calculus, though some require an understanding of probability (think maximum likelihood and Bayes' Rule) and basic linear algebra (think matrix operations and dot products). The appendix reviews the {doc}`math </content/appendix/methods>` and {doc}`probability </content/appendix/methods>`needed to understand this book. The concept sections also reference a few common machine learning {doc}`methods </content/appendix/methods>`, which are introduced in the appendix as well. The concept sections do not require any knowledge of programming. 

The *construction* and *code* sections of this book use some basic Python. The construction sections require understanding of the corresponding content sections and familiarity creating functions and classes in Python. The code sections require neither. 



## Where to Ask Questions or Give Feedback

You can raise an issue [here](https://github.com/dafriedman97/mlbook/issues) or email me at dafrdman@gmail.com. You can also connect with me on Twitter [here](https://twitter.com/dafrdman) or on LinkedIn [here](https://www.linkedin.com/in/daniel-friedman-36b1b2139/). 