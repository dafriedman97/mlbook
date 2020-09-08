# Concept

A *classifier* is a supervised learning algorithm that attempts to identify an observation's membership in one of two or more groups. In other words, the target variable in classification represents a *class* from a finite set rather than a continuous number. Examples include detecting spam emails or identifying hand-written digits. 

This chapter and the next cover *discriminative* and *generative* classification, respectively. Discriminative classification directly models an observation's class membership as a function of its input variables. Generative classification instead views the input variables as a function of the observation's class. It first models the prior probability that an observation belongs to a given class, then calculates the probability of observing the observation's input variables conditional on its class, and finally solves for the posterior probability of belonging to a given class using Bayes' Rule. More on that in the following chapter. 

The most common method in this chapter by far is logistic regression. This is not, however, the only discriminative classifier. This chapter also introduces two others: the *Perceptron Algorithm* and *Fisher's Linear Discriminant*.

