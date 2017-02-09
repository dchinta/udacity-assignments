#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','deferred_income','bonus','shared_receipt_with_poi','fraction_total_messages_poi'] -submitted 

#features_list=['poi','to_messages','deferral_payments','expenses','deferred_income','long_term_incentive','shared_receipt_with_poi','loan_advances','from_messages','director_fees','bonus','total_stock_value','from_poi_to_this_person','from_this_person_to_poi','fraction_total_messages_poi','restricted_stock','salary','total_payments','exercised_stock_options','income_fraction'] 
#features_list = ['poi','restricted_stock', 'total_stock_value', 'fraction_total_messages_poi', 'exercised_stock_options', 'from_this_person_to_poi', 'deferred_income', 'income_fraction']
#features_list = ['poi','total_stock_value', 'exercised_stock_options', 'from_this_person_to_poi', 'deferred_income', 'fraction_total_messages_poi', 'income_fraction']
#features_list = ['poi','exercised_stock_options','deferred_income','from_this_person_to_poi','fraction_total_messages_poi','income_fraction']
features_list = ['poi', 'fraction_total_messages_poi', 'deferred_income', 'from_this_person_to_poi','income_fraction']


# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
print "No. of data points in the dataset is ", len(data_dict)

# Looking at data by features by POI and non-POI classes
def plotfeatures(a,b):
    for key in data_dict:
        if (data_dict[key][a]!= 'NaN') & (data_dict[key][b]!= 'NaN'):
            if data_dict[key]['poi']:
                plt.scatter(data_dict[key][a],data_dict[key][b],color='r')
            else:
                plt.scatter(data_dict[key][a],data_dict[key][b],color='b')

    plt.xlabel(a)
    plt.ylabel(b)
    label=str(a+"_"+b)
    plt.savefig(label)
    plt.show()
    
#plotfeatures('salary','total_payments')

### Task 2: Remove outliers 

data_dict.pop('TOTAL', 0) 
### Task 3: Create new feature(s)
for each in data_dict.keys():
    if (data_dict[each]['from_messages']!='NaN')&(data_dict[each]['to_messages']!='NaN')&(data_dict[each]['from_poi_to_this_person']!='NaN')&(data_dict[each]['from_this_person_to_poi']!='NaN'):
        data_dict[each]['fraction_total_messages_poi'] = (data_dict[each]['from_poi_to_this_person']+data_dict[each]['from_this_person_to_poi'])/float(data_dict[each]['from_messages']+data_dict[each]['to_messages'])
    else:
        data_dict[each]['fraction_total_messages_poi'] = 0
        
for each in data_dict.keys():
    if (data_dict[each]['deferred_income']!='NaN')&(data_dict[each]['expenses']!='NaN')&(data_dict[each]['shared_receipt_with_poi']!='NaN'):
        data_dict[each]['income_fraction'] = (data_dict[each]['deferred_income']*-0.001+data_dict[each]['expenses'])/float(data_dict[each]['shared_receipt_with_poi'])
    else:
        data_dict[each]['income_fraction'] = 0        

# plotting the new feature 
#plotfeatures('salary','fraction_total_messages_poi')
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.pipeline import Pipeline


features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)


from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.grid_search import GridSearchCV 
# Decision Tree classifier
clf_dtc= DTC(min_samples_split=2, max_depth=3, min_samples_leaf=1)
# Naive Bayes 
clf_nb=GaussianNB()
# SVM after feature transformation via PCA 

estimators = [('scaler', MinMaxScaler()),('reduce_dim', PCA()), ('svm', svm.SVC())]
clf_svm = Pipeline(estimators)



#print clf.best_params_
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


params = {'min_samples_split':range(2,20), 'max_depth':range(1,10)}
dtc = DTC()
clf_dtc_tuned = GridSearchCV(dtc, params)


estimators = [('scaler', MinMaxScaler()),('reduce_dim', PCA()), ('svm', svm.SVC())]
params = dict(reduce_dim__n_components=[4,5,6],svm__C=[0.1,10,100])
pipe = Pipeline(estimators)
clf_svm_tuned = GridSearchCV(pipe, param_grid=params)

clf=clf_nb

#print clf.best_params_
#clf = DTC(min_samples_split=2,max_depth=2)
#clf=DTC()


# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.cross_validation import KFold
kf=KFold(len(labels),2)
for train_indices, test_indices in kf:
    #make training and testing sets
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]

clf.fit(features_train, labels_train) 
pred = clf.predict(features_test)

print "Accuracy score:", accuracy_score(pred, labels_test)
print "Precision score:", precision_score(pred, labels_test)
print "Recall score:", precision_score(pred, labels_test)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )

dump_classifier_and_data(clf, my_dataset, features_list)