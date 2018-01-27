#!/usr/bin/python

import sys
import pickle
sys.path.append("./tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'expenses', 'exercised_stock_options', 'restricted_stock', 'bonus',
                 'long_term_incentive', 'director_fees', 'total_payments']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# We're only removing the "total" field
data_dict.pop('TOTAL', 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)

# {'min_samples_split': 2, 'min_samples_leaf': 8}
# 0.87
# Precision: 0.39433
# Recall: 0.16700
# dt_params = {'min_samples_split': [2,4,6,8,10], 'min_samples_leaf': [1,2,3,4,5,6,7,8,9,10]}
# dt = tree.DecisionTreeClassifier(random_state=0)

# {'min_samples_split': 10, 'n_estimators': 10, 'min_samples_leaf': 3}
# 0.89
# Precision: 0.53591
# Recall: 0.14550
# rf_params = {'n_estimators': [5, 10, 15, 20, 25, 30], 'min_samples_split': [2,4,6,8,10], 'min_samples_leaf': [1,2,3,4,5,6,7,8,9,10]}
# rf = RandomForestClassifier(random_state=0)

# grid = GridSearchCV(dt, dt_params)
# grid.fit(features_train, labels_train)
# print(grid.best_params_)
# print(grid.best_score_)


# clf = SVC(C=0.125, gamma=0.0125, kernel='rbf')
# clf = tree.DecisionTreeClassifier(random_state=0, min_samples_split=2, min_samples_leaf=8)
# clf = RandomForestClassifier(random_state=0, min_samples_split=10, n_estimators=10, min_samples_leaf=3)
clf = GaussianNB()
test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)