# Identify Enron Fraud Report

## 1. Question 1
Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

The goal of this project is to take the dataset that contains information on a subset of Enron employees, and predict if they were a person of interest in the fraud.

### 1.1 Total no. of records and features

The dataset has 145 records in total (the 146th record was a "TOTAL" row that also got ingested). There are 21 distinct features. The target variable is "poi". Out of 146 people, there are 18 persons of interests. Thus, there are 128 people who are NOT person of interests. As shared in the Udacity lecture by Kate, this list of 18 persons of interest is not complete. As per the `poi_names.txt` file, there are 35 PoIs in total, but not all of them are present in the dataset.

### 1.2 Odd features

The one odd thing I found were the features 'shared_receipt_with_poi', 'from_this_person_to_poi', and 'from_poi_to_this_person'. I believe they cannot be used as features, because you can know these details only if you already know who the PoIs are. So I'm excluding these features from all of my predictive analyses.

### 1.3 Missing values

The features that have the most amount of missing values are:

1. loan_advances: Only has 4 values out of 146
1. deferred_income: Has 49 values out of 146
1. deferral_payments: Has 39 values out of 146
1. director_fees: Has 17 values out of 146
1. restricted_stock_deferred: Has 18 values out of 146

### 1.4 Features used

The features that I decided to use, considering my initial assumption about their usefulness, and the total no. of values present were:

1. director_fees
1. long_term_incentive
1. exercised_stock_options
1. restricted_stock
1. salary
1. bonus
1. expenses

The director_fees is the most interesting, since it has 17 records, and we have 18 PoIs. Although we'd normally be wary of a variable with so many NAs, this one could prove useful.

### 1.5 Outliers

1: There was an entry called "TOTAL", which had the total of all the columns for all the people. This row, thus, was an outlier for all the columns, and had to be removed.

![TOTAL outlier](images/salary_outlier.png "TOTAL Outlier")

2: There were 3 outliers for salary, namely "FREVERT MARK A", "LAY KENNETH L", and "SKILLING JEFFREY K". These are valid outliers, because they were involved in the fraud, and were holding important positions in the company. These outliers are not like the "TOTAL" one that we found earlier.

![Salary outliers](images/salary_bonus_outlier.png "Salary outliers")

![Salary outlier names](images/salary_bonus_outlier_names.png "Salary outlier names")

3: There was one outlier for bonus, "LAVORATO JOHN J" (as seen from the scatterplot above). Again, these are most likely indicators of PoIs, and not an outlier due to data entry error.

![Bonus outlier names](images/bonus_outlier_names.png "Bonus outlier names")

## 2. Question 2
What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]

### 2.1 Select features based on intuition

The following features are derived from PoIs:

1. shared_receipt_with_poi
1. from_this_person_to_poi
1. from_poi_to_this_person

However, to generate these features, you have to know beforehand whether the given person is a PoI. This looks like cheating, and I'm going to discard these features in my analysis. We're trying to predict the PoI field, so we shouldn't derive features from it. (Yes, we're tallying email address as persons of interest, but it still feels that defeating the purpose of machine learning)

I believe the following features could be good indicators of PoIs:

1. director_fees
1. long_term_incentive
1. exercised_stock_options
1. restricted_stock
1. salary
1. bonus

The reason is, the fraud was all about money. The people who made the most amount of money are highly likely to be persons of interest. I've avoided aggregate variables like total_payments because they could be redundant.

### 2.2 Come up with a new feature

Let's look at the plot below:

![Total payments vs. Salary - expense](images/salary_surplus.png "Total payments vs. Salary - expense")

I'm considering `salary - expense` to be a new feature. Generally, the higher the salary you draw the greater the expense. But I observed that in a few cases, the salary is disproportionately high compared to the expenses. Maybe this could catch a few PoIs, and I'll also need to only include this one variable `salary - expense` instead of salary and expense individually.


### 2.3 Getting rid of features that we don't want

The following features are the ones with a lot of NAs:

1. loan_advances: Only has 4 values out of 146
1. deferred_income: Has 49 values out of 146
1. deferral_payments: Has 39 values out of 146
1. director_fees: Has 17 values out of 146
1. restricted_stock_deferred: Has 18 values out of 146

I'd probably not use most of them, except for director_fees, which I think might help in identifying PoIs. There could be other variables from the above list that'd be helpful too, but I'm just picking one for now.

One correlated feature that I came across was `total_stock_value` and `exercised_stock_options`. You can't exercise more stock options than you have, so we see an almost linear relation. Thus, we only need one of the two features, which can reduce training time.

![Total stock value vs. Exercised stock options](images/stock_options_vs_exercised.png "Total stock value vs. Exercised stock options")

Also, there is one argument against ignoring features that have a lot of missing values. What if, by the virtual of values being missing or not, the target class can be predicted? What if loan_advances were only present for PoIs, and not for others? I agree, it could be. However, I'd still want values for everyone else, even if it's 0. Or else, these missing values could very well represent an external bias or suppositions by the authorities.

### 2.4 Univariate feature selection

I explore the selection of features using SelectKBest. I explored the following features:

1. director_fees
1. long_term_incentive
1. exercised_stock_options
1. restricted_stock
1. salary
1. bonus
1. expenses

My intention was not to find to top "K" features. My intention was to check how important each of them were. To do that, I ran SelectKBest with increasing values of k, from 1 to 5. I got the feature importances in this order:

1. exercised_stock_options
1. bonus
1. salary
1. long_term_incentive
1. restricted_stock

And it makes sense. The stock options and bonuses could have contributed more to the fraud than just pure salary.

## Question 3
What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?

I initally played around with SVC and DecisionTree in my jupyter notebook. However, after the train test split, SVC was just taking too long for the GridSearch, so I abondoned it.

I tried RandomForestClassifier, DecisionTreeClassifier, and GaussianNB.

**RandomForestClassifier:**
The best params were: {'min_samples_split': 10, 'n_estimators': 10, 'min_samples_leaf': 3}
The best score was: 0.89
Precision: 0.53591
Recall: 0.14550

**DecisionTreeClassifier:**
The best params were: {'min_samples_split': 2, 'min_samples_leaf': 8}
The best score was: 0.87
Precision: 0.39433
Recall: 0.16700

**GaussianNB:**
Accuracy: 0.80800
Precision: 0.34452
Recall: 0.48750

I did a stratified train-test split, and used GridSearchCV. However, in both RandomForestClassifier and the DecisionTreeClassifier, I could not reach 0.3 for both precision and recall. Surprisingly, GaussianNB gave me good results.

Thus, I decided to go ahead with GaussianNB.

## Question 4
What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

Tuning the hyperparameters of an algorithm helps with increasing accuracy of the classifier. I tuned using GridSearchCV, though in the end, I had to use GaussianNB which does not have tunable hyperparameters.

But I still ended up doing the tuning while trying out Random Forests and Decision Trees. You have to provide a list of possible values to try out for each parameter to GridSearchCV. Then, GridSearch will go through all the combinations of hyperparameters, and then run the classifier through K-Fold cross validation, and then tell you what set of parameters performed the best, and what the best score was.


## Question 5
What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]

If we train a classifier, and only look at the training performance, then we won't know if the classifier has really learnt the data or has just memorized the training set.

We should always measure the performance of the classifier on data that it has not seen before, so that we get an idea of how the classifier generalizes.

We thus divide the training data into a "training set" and a "validation set". The validation set is the set that the classifier does NOT get to train on. Once trained on the training set, we check the performance of the classifier on the validation set. If the classifier performed well on the training set, but not well on the validation set, then we know that the classifier is overfitting.

Validation helps us avoid overfitting. I used GridSearchCV for K-fold cross validation. However, since I went ahead with GaussianNB, which didn't need the GridSearchCV, performance was measured on the "test" set. Test set consisted of 30% of the data that the classifier did not train on, so that we could measure its performance on unseen data.

## Question 6
Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

The two metrics that I tracked were precision and recall. The final algorithm achieved a precision of 0.34452 and a recall of 0.48750.

Precision means, out of all the people we classified as persons of interest, how many were actually persons of interest. If we falsely flagged someone as a person of interest, then the precision reduces.

Recall means, out of all the people who are really persons of interest, how many could be identify as persons of interest. Could we "recall" that these people are persons of interest. If we fail to identify a person as a PoI, then recall of the classifier reduces.

