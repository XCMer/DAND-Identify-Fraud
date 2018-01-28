#!/usr/bin/python

import sys
import pickle
sys.path.append("./tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
use_salary_minus_expenses = True
features_list = ['poi', 'exercised_stock_options', 'restricted_stock', 'bonus',
                 'long_term_incentive', 'director_fees', 'total_payments', 'restricted_stock_deferred']
if use_salary_minus_expenses:
    features_list.append('salary_minus_expenses')
else:
    features_list += ['salary', 'expenses']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# We're only removing the "total" field
data_dict.pop('TOTAL', 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# Add the salary-expenses feature (the feature that I created)
for name, feature in my_dataset.items():
    if feature['salary'] == 'NaN' or feature['expenses'] == 'NaN':
        feature['salary_minus_expenses'] = 'NaN'
    else:
        feature['salary_minus_expenses'] = feature['salary'] - feature['expenses']

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
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.base import clone

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

# Construct the different pipelines that we want to test
pipelines = {
    'nb': Pipeline([
            ('skb', SelectKBest()),
            ('clf', GaussianNB())
        ]),
    'dt': Pipeline([
            ('skb', SelectKBest()),
            ('clf', tree.DecisionTreeClassifier(random_state=0))
        ]),
    'rf': Pipeline([
            ('skb', SelectKBest()),
            ('clf', RandomForestClassifier(random_state=0))
        ]),
}

# Construct parameters for each of the pipeline
param_grids = {
    'nb': {
        # We don't want to include "poi", but range function is not
        # inclusive of the upper bound. So we don't need to do a +1 in the upper
        # bound here.
        'skb__k': range(1, len(features_list))
    },
    'dt': {
        'clf__min_samples_split': [2,4,6,8,10],
        'clf__min_samples_leaf': [1,2,3,4,5,6,7,8,9,10],
        'skb__k': [1,2,3,4,5,6]
    },
    'rf': {
        'clf__n_estimators': [5, 10, 15, 20],
        'clf__min_samples_split': [2,4,6,8,10],
        'clf__min_samples_leaf': [1,2,3,4,5,6,7,8,9,10],
        'skb__k': [1,2,3,4,5,6]
    }
}

for classifier in ['dt', 'nb', 'rf']:
    # Do a grid search on the classifier params
    print "Classifier: ", classifier
    grid = GridSearchCV(pipelines[classifier], cv=StratifiedKFold(), param_grid=param_grids[classifier])
    grid.fit(features_train, labels_train)
    print "Best params: ", grid.best_params_
    print "Best score: ", grid.best_score_
    
    support = grid.best_estimator_.named_steps['skb'].get_support()
    support_names = []
    for index, s in enumerate(support):
        if s:
            # We do a +1 because the first feature is poi, which we don't train on
            support_names.append(features_list[index + 1])

    print "Best features: ", support_names

    # Now, test the classifier's best params against the test_classifier
    # poi has to be first, and we'll only select features returned by SelectKBest
    final_features = ['poi'] + support_names
    print "Final features list: ", final_features

    # Get params for the classifier into a dict
    clf_params = {}
    for key, value in grid.best_params_.items():
        if key.startswith('clf__'):
            # Remove the starting clf__
            clf_params[key[:5]] = value

    # Clone a new classifier instance and set its paramters
    new_clf = clone(grid.best_estimator_.named_steps['clf'])
    # new_clf = tree.DecisionTreeClassifier(random_state=0)
    for key, value in clf_params.items():
        setattr(new_clf, key, value)


    # Test the classifier
    test_classifier(new_clf, my_dataset, final_features)

# clf = SVC(C=0.125, gamma=0.0125, kernel='rbf')
# clf = tree.DecisionTreeClassifier(random_state=0, min_samples_split=2, min_samples_leaf=8)
# clf = RandomForestClassifier(random_state=0, min_samples_split=10, n_estimators=10, min_samples_leaf=3)
# clf = GaussianNB()
# test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

# The reason I had to modify the "features_list" here was because I'm only using a subset
# mentioned above. If I modified features_list with only the features that I'm using, I wouldn't
# have been able to use my grid search.
clf = GaussianNB()
dump_classifier_and_data(clf, my_dataset, ['poi', 'exercised_stock_options'])