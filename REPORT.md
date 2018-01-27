# Identify Enron Fraud Report

### Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

The goal of this project is to take the dataset that contains information on a subset of Enron employees, and predict if they were a person of interest in the fraud.

#### Total no. of records and features

The dataset has 145 records in total (the 146th record was a "TOTAL" row that also got ingested). There are 21 distinct features. The target variable is "poi". Out of 146 people, there are 18 persons of interests. Thus, there are 128 people who are NOT person of interests. As shared in the Udacity lecture by Kate, this list of 18 persons of interest is not complete. As per the `poi_names.txt` file, there are 35 PoIs in total, but not all of them are present in the dataset.

#### Odd features

The one odd thing I found were the features 'shared_receipt_with_poi', 'from_this_person_to_poi', and 'from_poi_to_this_person'. I believe they cannot be used as features, because you can know these details only if you already know who the PoIs are. So I'm excluding these features from all of my predictive analyses.

#### Missing values

The features that have the most amount of missing values are:

1. loan_advances: Only has 4 values out of 146
1. deferred_income: Has 49 values out of 146
1. deferral_payments: Has 39 values out of 146
1. director_fees: Has 17 values out of 146
1. restricted_stock_deferred: Has 18 values out of 146

#### Features used

The features that I decided to use, considering my initial assumption about their usefulness, and the total no. of values present were:

1. total_payments
1. total_stock_value
1. exercised_stock_options
1. restricted_stock
1. salary
1. bonus

#### Outliers

1: There was an entry called "TOTAL", which had the total of all the columns for all the people. This row, thus, was an outlier for all the columns, and had to be removed.

![TOTAL outlier](images/salary_outlier.png "TOTAL Outlier")

2: There were 3 outliers for salary, namely "FREVERT MARK A", "LAY KENNETH L", and "SKILLING JEFFREY K". These are valid outliers, because they were involved in the fraud, and were holding important positions in the company. These outliers are not like the "TOTAL" one that we found earlier.

![Salary outliers](images/salary_bonus_outlier.png "Salary outliers")

![Salary outlier names](images/salary_bonus_outlier_names.png "Salary outlier names")

3: There was one outlier for bonus, "LAVORATO JOHN J" (as seen from the scatterplot above). Again, these are most likely indicators of PoIs, and not an outlier due to data entry error.

![Bonus outlier names](images/bonus_outlier_names.png "Bonus outlier names")



