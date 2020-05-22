

#load geneal libraries
import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
import sklearn.metrics as metrics
from sklearn.naive_bayes import GaussianNB
import random

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")

# In[4]:


#init settings
seed = 309
random.seed(seed)
np.random.seed(seed)

#load data
data = pd.read_csv("adult.data", names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"], na_values="?")
test_set = pd.read_csv("adult.test",skiprows=[0], names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"], na_values="?")
#Preprocessing steps
# Encode the categorical features as numbers
def convert(df):
    result = df.copy()
    temp = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            temp[column] = preprocessing.LabelEncoder()
            result[column] = temp[column].fit_transform(result[column])
    return result, temp

train, _ = convert(data)
test, _ = convert(test_set)

#delete education - since it is the same as edu-num
del train['Education']
del test['Education']

#create dummy variables for categorical data
def createDummy(data):
        cat_vars = ['Workclass', 'Martial Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country']
        for var in cat_vars:
                cat_list = 'var' + '_' + var
                cat_list = pd.get_dummies(data[var], prefix=var)
                new_data = data.join(cat_list)
                data = new_data

        data_vars = data.columns.values.tolist()
        to_keep = [i for i in data_vars if i not in cat_vars]

        data_final = data[to_keep]
        data = data_final
        return data

train = createDummy(train)
test = createDummy(test)

# Get missing columns(from dummy variables) in the training test
missing_cols = set( train.columns ) - set( test.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
test = test[train.columns]
print train.shape

train = train.drop_duplicates()

print train.shape



#split train set by features and target
train_Y = train['Target']
del train["Target"]
train.head()

#split train set by features and target
test_Y = test['Target']
del test["Target"]
test.head()

#standardisation
from sklearn.preprocessing import StandardScaler

# Get column names first
names = train.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(train)
train = pd.DataFrame(scaled_df, columns=names)

scaled_df1 = scaler.fit_transform(test)
test = pd.DataFrame(scaled_df1, columns=names)


def eval(model, model_name):
        print '-------------------', model_name ,'---------------'
        # Train the model using the training sets
        model.fit(train,train_Y)
        #Predict Output
        predicted= model.predict(test)

        print(classification_report(test_Y, predicted))
        print 'AUC: ' ,metrics.roc_auc_score(test_Y, predicted)
        print 'Accuracy_score: ', metrics.accuracy_score(test_Y, predicted)
        print 'Precision: ', metrics.precision_score(test_Y, predicted)
        print 'Recall: ', metrics.recall_score(test_Y, predicted)
        print 'f1: ', metrics.f1_score(test_Y, predicted)


knn = KNeighborsClassifier()
eval(knn, "KNN")
gnb = GaussianNB()
eval(gnb, "Naive Bayes")
SVMC = svm.SVC(gamma='scale')
eval(SVMC, "SVM")
dt = tree.DecisionTreeClassifier()
eval(dt, "Decision Tree")
rf = RandomForestClassifier(n_estimators=100)
eval(rf, "Random Forrest")
ab = AdaBoostClassifier()
eval(ab, "Ada Boost")
gb = GradientBoostingClassifier()
eval(gb, "gradient boost")
ld = LinearDiscriminantAnalysis()
eval(ld, "Linear Discrimnant")
mlp = MLPClassifier()
eval(mlp, "MLP")
lr = LogisticRegression()
eval(lr, "Logistic Regression")