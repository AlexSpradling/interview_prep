
# Interview Prep I. 

## Statistics & Machine Learning
---
### What is the difference between unsupervised and supervised learning? **(repeat)**

Supervised learning models use *labeled training data* to make predictions. in linear regression, which is a supervised learning algorithm, you have a known output(label), and you use it to train a model to predict the output based on a set of input features. for example, you might use the square footage of a home as an input feature and the sales price as the label to train the model. 

On the other hand, unsupervised learning models don't have any labeled training data and instead try to find patterns or relationships within the input data itself. 

Some examples of supervised models:

* linear regression
* logistic regression
* tree classifiers (decision trees, random forest)
* k-nearest neighbors
* naive bayes

Some examples of un-supervised models:
* k-means clustering
* neural networks

---

### what is the difference between a regression problem and classification problem? provide an example of each. **(repeat)**

a regression problem involves predicting a *continuous output*, such as the price of a home or a quantity. an example of a regression problem is predicting the price of a home based on its square footage and lot size. 

a classification problem involves predicting a categorical output, such as some label or class. an example would be determining whether an image contains a hot dog or not, or whether a patient has a certain illness or not based on their symptoms and medical history. 

---

### what are the assumptions of a linear regression model? **(repeat)**

There are 5 main assumptions in a linear model. remember it as linem:

l inearity
i ndepdence of errors
n ormally distributed 
e quality of variance (homoscedasticity)
m ulticollinearity (none!)

1. *linearity* there is a linear relationship between the dependent variable and the independent variable. this means that the change in the dependent variable is proportional to the change in the independent variables.  

tests - residuals vs. fits: our residuals on the y-axis and our predicted values on the x. *if the residuals stay close to 0 as we scan the plot from left to right we're good*


2. *independence* the residuals are independent of one another, means that the residuals for one observation should not be related to the residuals for another observation. This assumption is important for ensuring that the model parameters are estimated correctly.

tests - durbin watson test (score of 2 we're good)

3. *normality* the residuals (the difference between the actual and predicted values) are normally distributed. 

tests - plot a histogram of residuals or use a qq-plot. 

4. *equality of variance (homoscedasticity)* the residuals have constant variance, meaning that the variance of the residuals is the same for all values of the independent variable. put another way, the size of the error term is constant across all ranges of the independent value. 

(example to explain more easily: if we were studying family income and luxury spending and studied the residuals we would see a problem, some families with lots of income spend a lot on luxury items while others do not -- the size of the error term is not constant across all ranges of the independent variables, a residuals vs. fits plot would show a cone shape!)

tests - residuals vs. fits: our residuals on the y-axis and our predicted values on the x. *if there is no pattern in the distribution of the residuals and they are equally spread around the line y = 0, we're good*


5. *lack of multicollinearity* there is no high correlation between the independent variables. this means that one independent variable is not a linear combination of the other independent variables.

Tests - Variance Inflation Factor.

$$
VIF = \frac{1}{1-R^2}
$$

vif is calculated by running a slr between any one feature x_i and the set of the rest, calculating the tolerance (1-r^2), and then taking the inverse of that value. the smaller the tolerance (the greater the r^2 score), the stronger the multicollinearity. 


---

### explain linear regression from a point of view that our ceo could understand. **(repeat)**

linear regression is a way to model the relationship between some continuous value or values: our independent variables, and some continuous value we would like to predict, our dependent variable.

we do this by fitting a line through the data and from that line we find the mathematical relationship between each independent variable and the dependent variable. the coefficient (slope of the fitted line) of the independent variables represents how much the dependent variable changes for a one unit increase in the independent variable, all other variables held constant. 

---

### how would you typically handle missing data? **(repeat)**

---
  
### let's say you build a model that performs poorly. how would you go about improving it? **(repeat)**

---

### how would you evaluate a classification model? what metrics other than accuracy might you use, and why? **(repeat)**

I would start with the confusion matrix and look at tp, tn, fp, fn.


quick review:

true positive = the class was a 1 and we predicted a 1.

true negative = the class was a 0 and we predicted a 0.

false positive = the class was a 0 and we predicted a 1. 

false negative = the class was a 1 and we predicted a 0. 


I think the importance of these metrics is a tailored a bit to the context of the classification problem. for example if its absolutely important we correctly classify all true positives, like if we're looking for ebola cases or something similar, we can lower the classification threshold and capture all of the actual positives but we'll get more false positives.

From the confusion matrix its easy to calculate some of the other classification metrics, suchs as:

sensitivity - how well our model classifies actual positives. 

tp/tp+fn.

specificity - how well our model classifies actual negatives. 

tn/tn+fp. 

precision - the percentage of predicted positives.

"pppp!"

tp/tp+fp

false positive rate - the percentage of actual negatives *misclassified* by our model. 

fp/fp+tn

I'd also want to look at the roc/auc curves for the model, sometimes they're a bit more helpful than the above stats. 

The roc curve is a graphical representation of how a binary classifier performs as the threshold is changed. it shows this by plotting the true positive rate (sensitivity) vs. the false positive rate. the auc (area under the curve) of the roc curve is a single scalar metric where 1 is perfect and .5 is random guessing. 

---

### what is better, false positives or false negatives? why? **(repeat)**

the answer really depends on the context. if we were trying to classify something like cases of ebola virus using temperature, we would want to set the classification threshold very low because it would be essential that we captured all of the true positive cases. as a result, we would get a number of false positives, but that would be preferable to false negatives. 

on the other hand, if we were classifying spam emails, we would would be more willing to accept false negatives (spam emails incorrectly classified as not-spam) since it would be more harmful to automatically delete important emails. 

---

### if a client who is not a math/data person were to ask you what are mcmc methods, how would you describe it to them? **(repeat)**

mcmc stands for markov chain monte carlo. it's a way of sampling from a complex probability distribution that we might not be able to calculate exactly. 

think of it like drawing random samples from a hat full of numbers, where the numbers have different weights - so some numbers are more likely to be drawn than others. in mcmc, we use a simulation to generate a set of samples from the hat, so that we can estimate the underlying probability distribution. 

bottom line: mcmc is a way of generating random samples that are representative of a complex distribution, and then using those samples to estimate the characteristics of the distribution. 

---

### how do you select the right model for a problem? (note: my wording here is verbatim. i asked for clarification about "what kind of problem" and they said "any problem." this forced me to describe a few different approaches.) **(repeat)**

---

### what is the bias-variance tradeoff? **(repeat)**

the bias-variance tradeoff is a principal in machine learning. the main thrust of it is that we want to train a model such that it has genuine predictive power and also generalizes to unseen data. 

a model with high-bias does not fit the training data perfectly. maybe its a simple linear regression. it doesn't capture all of the turns and non-linearity in the data, so we say it underfits the data. 

a model with high-variance will fit the training data really well, maybe its a complex multiple linear regression with lots of polynomial features and it captures every twist in turn in the training data, but then on unseen validation data it performs very poorly. we can't say for sure how it'll perform so it has *high variance*. in this case we say that high-variance models overfit the training data. 

the ideal model splits the difference between the two. 

---

### how do you correct for variance in a regression problem? **(repeat)**

variance in a regression problem means that the model is fit too closely to the noise in the training data and is overfit. this is going to result in poor predictive power on unseen data. 

some approaches to correct include:

1. regularization methods such as ridge, lasso, and elastic net, add a penalty term to the loss function being optimized in the model. this term encourages the model to have smaller coefficients, reducing the complexity of the model and decreasing the variance. 

2. dimensionality reduction via pca. 

---

### explain pca to me. **(repeat)**

pca stands for principal components analysis, its a method used to simplify the structure of large and complex datasets. it reduces the number of features in a data set by combining related features into a new set of derived features called *principal components*, which capture most of the variation in the data. 

it does this by transforming the original feature space into a new feature space, which is a lower-dimensional space, where the first principal component is the most important and has the largest variation in the data, the second principal component has the second largest variation and so on.

the idea is that is we project our original data set onto this new feature space, it captures most of the variation in the data, but with fewer features, which makes it easier to analyze and visualize.

---

### what is the difference between a decision tree and a random forest? **(repeat)**

a decision tree is a tree like structure, where the internal nodes are the features or predictors of the data, and the leaves represent the class labels or target variable. 

here's how a decision tree works:

1. the goal of building a decision tree is to find the feature that best splits the data into groups with similar target values. the first step is calculate the gini impurity for each feature. gini impurity is a measure of how well a feature separates the data into different target classes. it ranges from 0 (perfect separation) to 1 (no separation). we can calculate it like this:

gini = 1 - p(yes)^2 - p(no)

where p(y/n) is the number of y or no in the leaf / the total # in the leaf.

2. the feature with the lowest gini impurity is selected as the root node of the tree. this feature is then used to split the data into two groups based on whether the feature value is above or below a certain threshold. for example, if the feature is "age", the two groups might be < 40 and > 40. 

3. for each of the two groups created in the previous step, the process is repeated by calculating the gini impurity for each feature and selecting the feature with the lowest value as the next node in the tree. this process continues until all nodes are pure -- meaning that all the data points withing a node belong to a single class. at this point, the tree is complete, and new data can be classified by following the tree from the root node to a leaf. 

a random forest is an extension of a decision tree. instead of just having one decision tree, a random forest is made up of many decision trees. the idea is that the more decision trees you have, the better your model will be at generalizing to new data. 

each tree in a  random forest is grown independently from a random subset of the training data and from a random subset of the features. the idea is that each tree will be different because it's grown from different data and different features. at the end of the day, when we make a prediction, we make a majority vote from all the trees int eh forest, and the prediction that appears most frequently wins. 


### what is boosting? **(repeat)**

---

### what is machine learning?

---

### what is a p-value?

a p-value is a measure of evidence against the null hypothesis in a hypothesis test. it represents the probability of observing a test statistic as extreme or more extreme than the one calculated from the sample, given that the null hypothesis is true. if the p-value is less than the chose significance level (.05), it suggests that the observed data is inconsistent with the null hypothesis and supports the alternative hypothesis. 

---

### what is a standard deviation?

standard deviation is a statistical measure of the spread or dispersion of data around the mean. it is calculated by finding the variance of the data and taking the square root of the variance. the variance is calculated as the sum of squared differences between each data point and the mean, divided by either the sample size(n) or the sample size - 1 (n-1), depending on whether you are calculating the population standard deviation or the sample standard deviation. 


$$
\textrm{standard deviation} = \sqrt{\frac{\sum(x_i-\mu)^2}{n}}
$$

---

### how is standard deviation used to construct a confidence interval?

a confidence interval is used when we're trying to estimate a population parameter with a sample. 


$$
ci = \bar{x} \pm z \frac{s}{\sqrt{n}}
$$

where:

$\bar{x}$ = sample mean

$z$ = critical value

we get the critical value by first calculating $(1-\frac{\alpha}{2})$, where $\alpha$ is the chosen confidence level. we then look up the value in a z-table if n > 30 or a t-table if n < 30. 

$s$ = sample standard deviation

$n$ = sample size 

Reminder:

$$
\frac{s}{\sqrt n} = \textrm{Standard Error}
$$


---

### under the hood, how is a linear regression model created? assume there is only one x variable.

we need to fit a line to our data that minimizes the sum of squared residuals (ssr) that's $\sum(y_i - \hat{y})^2$

ordinary least squares is one of the only models that has an analytical solution and we don't need to use gradient descent to optimize our loss function. so the formula for linear regression is $y=\beta_0 + \beta_1 x$.

we solve for $\beta_1$ by first finding the standard deviation of x and the standard deviation of y, then we calculate $r$, the pearson correlation coefficient:

$$
r = \frac{cov(x,y)}{s(x)\times s(y)}
$$

where:

$$
cov(x,y) = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{n-1}
$$

and $s(x)$ = 

$$
\textrm{sample standard deviation} = \sqrt{\frac{\sum(x_i-\bar{x})^2}{n-1}}
$$

$$
\beta_1 = \frac{r \times s_y}{s_x}
$$

with \beta_1, $x$ and $y$ we can solve for the intercept, $\beta_0$

$$
\beta_o = \bar{y} - \beta_1 \bar{x}
$$

---

### can you run linear regression on both numerical and categorical features?

you can but you need to one-hot-encode the categorial features, which is just a fancy of saying converting the categorical feature into a column of 1's and 0's.

---

### how would you need to preprocess your target variable if it were a categorical variable assuming you want to run a regression model?

You would need to dummify your target variable, which means convert it to a 1 or a 0. 

---

### what are the assumptions of a simple linear regression model?

There are 5 main assumptions in a linear model. remember it as linem:

l inearity

i ndepdence of errors

n ormally distributed 

e quality of variance (homoscedasticity)

m ulticollinearity (none!)

1. *linearity* there is a linear relationship between the dependent variable and the independent variable. this means that the change in the dependent variable is proportional to the change in the independent variables.  

Tests - residuals vs. fits: our residuals on the y-axis and our predicted values on the x. *if the residuals stay close to 0 as we scan the plot from left to right we're good*


2. *independence* the residuals are independent of one another, means that the residuals for one observation should not be related to the residuals for another observation. This assumption is important for ensuring that the model parameters are estimated correctly.

tests - durbin watson test (score of 2 we're good)

3. *normality* the residuals (the difference between the actual and predicted values) are normally distributed. 

tests - plot a histogram of residuals or use a qq-plot. 

4. *equality of variance (homoscedasticity)* the residuals have constant variance, meaning that the variance of the residuals is the same for all values of the independent variable. put another way, the size of the error term is constant across all ranges of the independent value. 

(example to explain more easily: if we were studying family income and luxury spending and studied the residuals we would see a problem, some families with lots of income spend a lot on luxury items while others do not -- the size of the error term is not constant across all ranges of the independent variables, a residuals vs. fits plot would show a cone shape!)

tests - residuals vs. fits: our residuals on the y-axis and our predicted values on the x. *if there is no pattern in the distribution of the residuals and they are equally spread around the line y = 0, we're good*


5. *lack of multicollinearity* there is no high correlation between the independent variables. this means that one independent variable is not a linear combination of the other independent variables.

tests - Variance Inflation Factor.

$$
VIF = \frac{1}{1-R^2}
$$

vif is calculated by running a slr between any one feature x_i and the set of the rest, calculating the tolerance (1-r^2), and then taking the inverse of that value. the smaller the tolerance (the greater the r^2 score), the stronger the multicollinearity. 

---

### how do you investigate whether there are any relationships between your features and between your features and the target?

---

### what is the difference between r-squared and adjusted r-squared?

R-squared is the total amount of variance that is accounted for by our model. Adjusted R-squared is the total amount of variance accounted for in our model, but additionally accounts for the number of features we have in our model and penalizes R-squared value based on the number of features. We calculate $R^2$ and Adjusted $R^2$ like this:

#### $R^2$

1. Find the Residual Sum of Squares:
$$
RSS = \sum{(y_i-\hat{y})^2}
$$

2. Find the total sum of squares
$$
TSS = \sum{(y_i-\bar{y})^2}
$$

We then get $R^2$ from:

$$
R^2 = (1-\frac{RSS}{TSS})
$$

#### Adjusted $R^2$
$$R^2 = (1-\frac{\frac{RSS}{(n-k)}}{\frac{TSS}{n-1}})$$

---

### in a linear model, how would you interpret a q-q plot?

A qq plot is used to assess whether the residuals of a linear regression are normally distributed. In a QQ plot the x-axis represents the quantiles from a normal distribution and the y-axis represents the quantiles from the residuals. 

If the scatter plot resembles a straight line than this would indicate the observed residuals follow a normal distribution. If the residuals don't indicate a normal distribution, than we did to transform the features as we haven't met some linear assumptions.

---

### how would you convey the correlation between two variables to a non-technical audience?

Correlation is an indication of whether two variables move in the same direction or in opposite directions. For example if two variables are positively correlated we expect them to move together, for example home sales price and square footage both rise together. Whereas home sales price and the number of proximal industrial sights are negatively correlated, as the number of sites goes up, sales price goes down. 

---

### what is the correlation between x and x?

x and x are perfectly correlated, so the correlation is 1. 

---

### what are the bounds of correlation between any two variables?

The bounds of correlation are -1 and 1, where -1 is perfect negative correlation and 1 is perfect positive correlation. 

---

### what is the correlation between x and x^2?

Should be 0. x and x^2 don't share a linear relationship.

---

### how would we know if we have any multicollinearity in our model?

One common test for multicollinearity is to calculate variance inflation factor. 

$$
VIF = \frac{1}{1-R_i^2}
$$

Reminder that:

$$
R^2 = \frac{RSS}{TSS}
$$

What we do is calculate a multiple regression with each feature as the response and all of the other features as predictors. We do this for each feature and collect all of the VIF scores. A VIF of 1 indicates low multicollinearity whereas a VIF over 5 indicates high multicollinearity. 

---

### given summary statistics, which variable should we drop from our regression model so we don't have multicollinearity?

If we were given summary statistics i.e. IQR, standard deviation, mean, and median, it might be difficult to find a confounding variable that is introducing multicollinearity. Summary statistics are going to be helpful to understand the spread, distribution, and central tendency of the data. 


VIF is a better indicator. 

---

### suppose you have an x variable that impacts y differently depending on *another* x variable. (for example: the dosage of a medication impacts your overall recovery time differently if you are a man or woman.) how do you account for this in a linear regression?

The answer is going to lie in interaction terms...So this is creating a new feature that is the product of two existing features. In this case we would create an interaction term between "dosage" and "gender".

---

### how can you assess whether your model is overfitting?

We're going to want to split our data into training and testing segments and evaluate the scores from each group. If the training score is high and the testing score is low we've definitely overfit our model. The corect way to compensate for this is to use cross validation in the training process:

**Cross Validation**

$k$ fold cross-validation is accomplished by iteratively removing a section of the data (that's the holdout set) we train the model on the remaining subsets and then predict on the holdout. That's 1 fold, and we we do this until we have folded the data $k$ times. It's common to use 5 folds. In doing this, we ensure the model is tested on data it hasn't seen during training, and gives us a more accurate assessment of the model's performance on unseen data.   

---

### what types of regression models do you know and what are the differences among them?

1. Simple Linear regression

$$
y = \beta_0 + \beta_1 X_1
$$
2. Multiple Linear regression

$$
y = \beta_0 + \beta_i X_i \dots \beta_n X_n
$$

For all of the above we are fitting a line by minimizing RSS. Reminder:

$$
RSS = \sum(y_i - \hat{y_i})^2
$$

3. Regularized Multiple Linear Regression

Regularization applies a penalty term to the best fit derived from the training data so that we can achieve lesser variance, so we're changing the cost function a bit.

**Ridge**

Best suited for multicollinearity. Analytical solution is available.

Ridge adds a penalty to the loss function. The Ridge penalty is the sum of the squared $\beta$ values:


$$
\lambda_2 + \sum \beta_j^2
$$

Where $\lambda_2$ is a constant for the strength of the regularization parameter. The greater this value, the higher the impact of this new component of the loss function, so 0 = regular RSS.

**LASSO**

Best suited for feature selection -- it "zeroes out" variables. Analytical solution is not available -- have to use gradient descent. 

Lasso adds the sum of the absolute value of the $\beta$ coefficients. 

$$
\lambda_1 \sum |\beta_j|
$$

**Elastic Net**

We use both penalties!

---

### what is regularization? why would you use it?

Regularization is a special type of multiple linear regression where penalties are imposed on the RSS cost function. We do this because we might be interested in reducing multicollinearity or performing feature selection when we have many variables. There are 3 main types:

Reminder, least squares minimizes RSS:
$$
RSS = \sum{(y_i - \hat{y_i})^2}
$$
1. Ridge

Ridge adds the sum of the squared $\beta$ coefficients multiplied by the $\lambda_2$ penality to RSS.

$$
RSS + \lambda_2\sum(\beta_j^2)
$$

2. Lasso 

Lasso adds the sum of the absolute value of the $beta$ coefficients, multiplied by the $\lambda_1$ penalty to RSS.
$$RSS + \lambda_1\sum(|\beta_j|)$$

3. Elastic Net

Elastic net adds both penalties!

---

### how does logistic regression work?

Logistic regression uses the same formula as linear regression except we change the response variable using a sigmoid function:
$$
log(\frac{1}{1-p})
$$

So now we have:

$$
log(\frac{1}{1-p}) = \beta_0 + \beta_i X_1 \dots \beta_n X_n
$$

Where $p$ is the probability of $y$ equaling one. 

The sigmoid function bounds y to a domain of 0 and 1 and allows us to use logistic regression for binary classification problems, we could have used some other function but the sigmoid function makes the math work out nicely.  

**Finding the Curve that maximizes maximum likelihood**

We fit a bunch of curves to the data and select the curve that maximizes likelihood, which is the probability of y = 1, and we do this using gradient descent.

Once we have the curve that best fits the data, we set a classification threshold, for example 0.5. Then, when its prediction time we read our x-value vertically to the sigmoid curve and read across to the y-axix from the point that it touches the curve to recover the probability of y equaling 1, given the x value. If the probability is above our threshold, then we classify as 1. 

---

### how do decision trees work?

Decision trees divide the data into smaller groups based on the values of the features in the data. The goal is to create a tree of decision nodes that allows us to make predictions by following a series of yes or no questions. 
1. Given data, we need to figure out which feature is the root node of the decision tree, so calculate the Gini Impurity value for each datapoint:

$$
Gini = 1 - p(yes)^2 - p(no)^2
$$

Where $p(yes)$ and $p(no)$ = 

$$
\frac{Total(\textrm{yes or no})}{\textrm{Total of leaf}}
$$

Gini impurity determines the best feature to split the data into smaller groups at each decision node. It measures the probability of misclassifying a randomly chosen element in the dataset if it were randomly labeled according to the class proportions in the group. 

We do this for every point and choose the feature with the lowest Gini impurity as the root node. 

2. From the root node, we repeatedly split the data into smaller groups based on the feature values until we reach a stopping criteria, such as all nodes being pure (contain only elements from one class) or we reach a pre-set maximum depth limit. The process of repeatedly splitting the data  involves again calculating the gini impurity of the remaining features and using the features with the lowest gini impurity value as the new decision node. 

3. Once the tree is built, we assign output values based on a majority class vote in the leaves contained in each decision node. 

#### Silly but useful StatsQuest Example

Dataset

| Loves Popcorn | Loves Soda | Age | Target: Loves Troll 2 |
|---------------|------------|-----|-----------------------|
| Yes           | Yes        | 7   | No                    |
| yes           | No         | 12  | No                    |
| No            | Yes        | 18  | Yes                   |
| No            | Yes        | 35  | Yes                   |
| yes           | Yes        | 38  | Yes                   |
| yes           | No         | 50  | No                    |
| No            | No         | 83  | No                    |

Gini Impurity for Loves Soda:

$$
\begin{align*}
\textrm{Left Leaf Gini (loves troll 2)} = 1 - (\frac{3}{4})^2 - (\frac{1}{4})^2 = .375\\
\textrm{Right Leaf Gini(does not love troll 2)} = 1 - (\frac{0}{3})^2 - (\frac{3}{3})^2 = 0
\end{align*}
$$

Except the right leaf gini is using 3 data points instead of 4, so we take the weighted average of both:

$$
\frac{4}{4+3}0.375 + \frac{3}{4+2}0 = 0.214
$$

```
                                                                       
                                +------------+                         
                                | Loves Soda |                         
                                +------------+                         
                             Y    /-     -\    N                       
                                 -         -                           
                         Loves Troll 2    Loves Troll 2                
                                                                       
                         Y          N      Y         N                 
                         3          1      0         3                 
                                                                       
                             /                                         
                            /                                          
                          /-                                           
                 +-------------+                                       
                 | Age < 12.5 |
                 +-------------+                                       
                 Y   /    \   N                                        
                    -      -                                           
         Loves Troll 2     Loves Troll 2                               
                                                                       
         Y         N       Y         N                                 
         0         1       3         0
```

---

### if you're comparing decision trees and logistic regression, what are the pros and cons of each?

Pros of Logistic Regression
- Fast
- Coefficients are interpretable as the change in the log odds of the target variable given a unit increase in a predictor. This makes it easier to understand how the variables are related to the target variable. 

Cons 
- Assumes a linear relationship between the dependent and independent variables
- Does not perform well with multi-class classification problems


Pros of Decision trees
- Can handle non-linear relationships and are good choice for complex datasets.
- Easy to understand out the decisions are being made. 
- Can be applied to regression and classification problems  

Cons
- Prone to overfitting - captures a lot of noise in the data instead of real underlying relationships
- Accuracy can be lower

---

### what are the advantages of using a random forest over a logistic regression model?

---
### what is bootstrapping?

Random sampling with replacement, many times. The advantage of bootstrapping is that we can use it to estimate a sampling distribution from a relatively small dataset.

---


### when bootstrapping, how would you estimate the error between your estimate and the true population value as a function of your sample size? (this relies on the central limit theorem.)

The central limt theorem states that with a large enough sample size, the distribution of the sample means will approximate a normal distribution, with a mean equal to the population mean and a standard deviation equal to the population standard deviation divided by the square root of the sample size (this is called standard error - the variation in the means from multiple sets of measurements)


Reminder, Standard Error = 

$$
\frac{\sigma}{\sqrt{n}}
$$

Where $\sigma$ is the population standard deviation and $n$ is the sample size. 

In bootstrapping, we use repeated resampling to estimate the distribution of our statistic of interest, such as a mean. The standard deviation of this distribution is a measure of the error in our estimate. As the number of bootstrap samples increases, this distribution will become more tightly centered around the true population value, and the standard deviation will decrease, approaching the standard error that we would expect given the CLT. So, as the sample size increases, the error between the bootstrapped estimate and the true population value should decrease according to the square root of the sample size.  

Therefore, the error between the estimate and true population value can be estimated as $Error = z \times SE$, where $z$ is the critical value (1.96 for 95% confidence from a z-table)

---

### what is bagging?

Bagging stands for "bootstrap aggregating" and can improve the accuracy for a model. It works like this:

You create multiple versions of a model, and each version is trained on a random subset of (with replacement) of the training data. The final prediction is made by aggregating (by averaging or voting) the predictions of all of the models. The ideal is to curb over fitting. 

---

### why might a random forest be better than a decision tree? (note: what they were looking for is to talk about how random forests work.)

A decision tree is a flowchart of sorts where a cascade of yes or no questions allows you to map every twist and turn in the data and arrive at a classification or regression prediciton. 

A random forest is a collection of many decision trees, except the trees are created using random subsets of the training data. The idea is that many un-correlated trees perform better as an ensemble and are less prone to over fitting.

---

#### How Random Forests Work

1. We take a random sample with replacement from our data (a bootstrapped sample).
2. We create a decision tree from the bootstrapped sample, using Gini impurity to find the root node and the decision nodes. 
3. We repeat the first 2 steps many times, creating many trees. Each time we sample a new bootstrap sample and create a new decision tree using Gini impurity. 
4. Majority voting: We take a majority vote from the outputs of each decision tree to make a final prediction. 

Reminder: Gini Impurity = 

$$
1 - prob(yes)^2 - prob(no)^2
$$

Where:
$$
\begin{align*}
prob(yes) = \frac{yes's}{\textrm{total in node or split}}\\
prob(no) = \frac{no's}{\textrm{total in node or split}}
\end{align*}
$$

---


### what are some models you could use to solve a classification problem?

**Models I can explain right now**

- Logistic Regression
- Decision Trees
- Random Forests

(I think I've explained these models pretty well above)

**Models I can't explain well right now**

- Naive Bayes 
- Extra Trees
- KNN

---

### Why might accuracy not be the best error metric for classification? 

---

### Given a confusion matrix, calculate precision.

#### The Confusion Matrix


| *     | Yes              | No             |
|-------|------------------|----------------|
| Yes   | True Positive    | False Negative |
| No    | False Positive   | True Negative  |



$$
Precision = \frac{TP}{TP+FP}
$$


---

### what is the difference between precision and recall?

Precision is the proportion of predicted positives in a model. 

$$
Precision = TP / TP + FP
$$

Recall = Sensitivity and is the proportion of actual positives correctly classified by the model. 

$$
Recall = TP / TP + FN
$$

---

### what are precision, recall, f1 score, and auc roc? when would you use each of them?

precision is the proportion of predicted positives by the model.
$$
precision = TP / TP + FP
$$

Recall (aka Sensitivity) is the proportion of actual positives predicted by the model. 

$$
Recall = TP / TP + FN 
$$

F1 summarizes the balance between precision and recall and can be used as an overall measure of a model's accuracy. It is the harmonic mean of precision and recall and is bound between 0 and 1.

$$
F1 = 2 \times \frac{precision \times recall}{precision + recall}
$$

---

### how is the roc curve created? 

ROC cuve stands for "Reciever Operating Curve" and plots the True Positive Rate against the False Positive Rate at a variety of classification thresholds. The True Positive Rate is on the y-axis and the False Positive Rate is on the X-axis. We want to the curve to tuck as far as possible into the upper left corner. The worst case scenario is just a diagonal line across the curbe

Reminder:

Sensitivity is the proportion of True Positives correctly classified by the model. 

$$
True Positive Rate = Sensitivity = Recall = \frac{TP}{TP+FN}
$$

The False positive Rate is 1 - Specificity:

Where: 
$$
Specificity = \frac{TN}{TN+FP}
$$

---

### what is the difference between parameters and hyperparameters?

Parameters are values that are learned and optimized by the model during training. An easy example is the coefficients learned by linear regression. 

Hyperparameters are settings for the model that control the behavior of the learning algorithm - 

### if you were a machine learning model, what would you be and why?

ED-209.


### is there a model that you tend to use frequently in your projects?  please explain how that model works.

I'll probably say linear regression because I can explain it well and I like the interpretability of the results. 

### why might multiple random weak trees be better than one long tree? (note: this is the difference between boosting and a deep decision tree.)

### what is gradient boosting?

### what is stacking?

Stacking is an ensemble learning method that combines multiple machine learning models. The idea is that different models have different strengths and weaknesses and by combining their predictions we create a more robust and accurate model overall. 

Example:

Consider a binary classification problem:

1. First we train two base models: a decision tree, and a random forest. Each model generates a prediction. 

2. We combine the predictions of the 2 base models into a new feature set, which will be the input to a meta-model, we'll use logistic regression. 

3. We train the logistic regression meta-model on the combined predictions of the base models.

### what are the pros and cons of bagging, boosting, and stacking?

### would you use dimensionality reduction with a random forest or gradient boosted model? why or why not?

Sure. Dimensionality reduction can be used to improve the performance of either model by reducing the number of features in the data, reducing the risk of over fitting. We just have to use it with care as we might lose important information or interpretability.  


### what is the definition of gradient?

A fancy word for a derivative. A gradient is a vector that points in the direction of of greatest increase of a function. 

### given a gradient f'(x) = x + 8, what is the antiderivative f(x)?

$$
\begin{align*}
\textrm{Solve this:}\\
= \int (x + 8)dx\\
\\
\textrm{Use the Linearity Rule:}\\
= \int x \ dx + 8 \int 1 dx \\
\\
\textrm{First solve for:}\\
\int x \ dx\\

\textrm{Use the power rule of integration:}\\
\int x^n dx = \frac{x^{n+1}}{n+1}\\
= \frac{x^2}{2}\\
\textrm{Now solve for:}\\
\int 1 \  dx\\
\textrm{Apply constant rule:}\\
= x\\
\textrm{Plug in the solved integrals...}\\
\int x \ dx + 8 \int 1\  dx \\
= \frac{x^2}{2} + 8
\end{align*}
$$

### given a function, how do you find its maxima or minima?

1. Analytical solution if one exists, like in Linear Regression. 

2. Graphical Inspection

3. A numerical method like Gradient Descent or Newton's method, which involve iteratively updating an initial guess for the maxima or minima until a solution is found. 

### Big Explanation of Gradient Descent (Beware!)



### how is stochastic gradient descent different than gradient descent?

### why are missing values a concern in modeling?

### what are the common pitfalls when building a predictive model? how do you avoid them?

### given three columns containing some missing data (one categorical and two quantitative columns), how would you impute values for a fourth column, where this fourth column can take on only positive integer values?

### what is your favorite use case for a machine learning model?

### what is the difference between k-means and k-nearest neighbors?

### what clustering algorithms have you used before? can you think of a reason why we'd use a clustering algorithm for a specific business problem?

### if i were to ask you to use clustering on some of the features in your ames housing dataset which i see you used for a project, how would you preprocess the data before applying the clustering?

### in a level of technical detail that's most comfortable to you, describe to me how dbscan works and in what cases would you choose it over k-means.

### suppose you're trying to cluster your observations into groups. however, you have too many features in your data. how would you decide which ones to use in the clustering model?

### when would you use pca? 

### why would you perform pca even if you don't have a lot of features?

### in what cases would you not use pca?

### what is the geometric interpretation of an eigenvalue?

### what is the algebraic interpretation of an eigenvalue?

### how would you detect anomalies in a dataset? how would you deal with them?

### what do you know about topic modeling and how can you apply it to business problems?

### what are time series data?

### what are some disadvantages to time series analysis methods?

### when are rnns used?

### how do recurrent neural networks (rnns) work?

### what hyperparameters are available in lstms?

### explain neural networks to our ceo.

### explain convolutional neural networks (cnns) to me as if i do not know data science.

### let's say you're building a neural network model to classify images. you image dataset is very small (say only 5 images). you need more data to build a relatively reliable model but there's no place for you to get more data from. what would you do?

### you have a dataset of very few training examples. you have a much much bigger set of outside data for which you're trying to make predictions. how do you determine whether your small training set is large enough to build a model and reliably deploy it to a much larger outside set?

### how does backpropagation work in neural nets?

### what types of neural networks are you familiar with? 

### what is the relationship among cnns, rnns and lstms?

### what type of data are neural networks best for?

### what use cases are good applications of neural networks?

### why are cnns good for image data?

### what are the drawbacks of data augmentation?

### why would you choose to use nltk as opposed to other natural language packages?

### you mentioned cosine similarity. how would you explain this to a non-technical person?

### aside from the sum of the observations divided by the number of observations, how would you describe the mean? (answer: it is the quantity that minimizes the sum of squared errors.)

### aside from the middle observation in a dataset, how would you describe the median? (answer: it is the quantity that minimizes the mean absolute error.)

### how do you calculate the l1 distance between two vectors?

### what's the difference between statistics, machine learning and deep learning. what is ai?

### how do you calculate the maximum model lift if the prevalence of the model is set at 20%? ([helpful link here](https://en.wikipedia.org/wiki/lift_(data_mining)).)

### after testing several models, how do you decide which is the best?

### when processing text, what do you do about frequently occurring words, like "a," "and," "the," etc.?

### when you pulled data from an api, what format was it in?



