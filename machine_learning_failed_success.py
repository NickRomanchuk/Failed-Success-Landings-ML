################################################
##  IOC 2021  by N. Romanchuk Nov. 10th, 2021 ##
################################################
#####################
## Import Packages ##
#####################
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import imblearn
from numpy import mean
from numpy import var
from math import sqrt

######################
## Data Exploration ##
######################
random.seed(42)

#Change settings so we can see all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Load in the csv and store as a dataframe
data = pd.read_csv(r'D:/Elodie_Summer_2/DataSet_2.csv')

#Eliminate trials that are 'fuzzy'
data.drop(data.index[data['Blurry'] == 1], inplace=True)

#drop columns that we don't need
data.drop(['File','Blurry','Leg','Type'], axis=1, inplace=True)

#Determine the class imbalance
Imbalance = data['y'].value_counts()

#######################
## Data Partitioning ##
#######################
participants = data['Participant'].unique() #array that conatins the participant codes 
sex = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #array that indicates if par is female (1) or male (0)

#stratify random sample data into training and testing set
train, test = train_test_split(participants, test_size=0.2, stratify=sex, random_state=42) #divide the participants into testing and training sets, stratifing based on sex

data_train = data[pd.DataFrame(data.Participant.tolist()).isin(train).any(1).values] #divide the whole data set according to the participants selected for training and testing
data_test = data[pd.DataFrame(data.Participant.tolist()).isin(test).any(1).values]

#############################################
## Variable Elimination due to correlation ##
#############################################
#create correlation matrix (this will be used to eliminate variables that are correlated with each other)
X = data_train.drop(['Sex','Dominant','Participant','y'], axis=1) #drop categorical columsn
Correlation = abs(X.corr())

#for variables that are correlated, determine which has the greatest effect size
#function that calculates cohens d
def cohend(d1, d2):
	# calculate the size of samples
	n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
	s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
	# calculate the pooled standard deviation
	s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
	u1, u2 = mean(d1), mean(d2)
	# calculate the effect size
	return abs((u1 - u2) / s)

#drop variables that are correlated with each other >0.7, only keep the variable that has the largest effect size
data_train = data_train.drop(['Jump_Height','Hip_Inv_x','Hip_Inv_y','HAT_x','HAT_hypo','Body_x','Body_y','Vel_HAT_x','Vel_HAT_hypo','Vel_Body_x','Vel_HAT_y'], axis=1)
data_test = data_test.drop(['Jump_Height','Hip_Inv_x','Hip_Inv_y','HAT_x','HAT_hypo','Body_x','Body_y','Vel_HAT_x','Vel_HAT_hypo','Vel_Body_x','Vel_HAT_y'], axis=1)

##########################################################
## Normalize only the numeric variables in the data set ##
##########################################################
df_num = data_train.drop(['Sex','Dominant','Participant','y'], axis = 1)
df_norm = (df_num - df_num.mean()) / (df_num.std())
data_train[df_norm.columns] = df_norm

df_num = data_test.drop(['Sex','Dominant','Participant','y'], axis = 1)
df_norm = (df_num - df_num.mean()) / (df_num.std())
data_test[df_norm.columns] = df_norm

################################################
## One hot encoding for categorical variables ##
################################################
data_train['Sex'] = data_train['Sex'].eq('F').mul(1)
data_train['Dominant'] = data_train['Dominant'].eq('D').mul(1)

data_test['Sex'] = data_test['Sex'].eq('F').mul(1)
data_test['Dominant'] = data_test['Dominant'].eq('D').mul(1)

################################################################################
## select variables using RFE and LOOCV (were each left out is a participant) ##
################################################################################
CV_all = []
participants = data_train['Participant'].unique() #array that conatins the participant codes 

for i in range(1,21):
    print(i)
    cv_score = []

    for j in range(len(participants)): #loop through each participant where each one is 'left out' and used as the 'testing set'
        logreg = LogisticRegression(random_state=42, penalty='none')
        rfe = RFE(logreg, n_features_to_select=i) #initalize RFE requesting that it find the best 'i' number of variables
        
        Par_train = np.delete(participants, (j)) #list of participants to train model
        Par_test = [participants[j]] # list of participants to test model

        data_valid = data_train[pd.DataFrame(data_train.Participant.tolist()).isin(Par_train).any(1).values] #divide the whole data set according to the participants selected for training and testing
        data_valid_test = data_train[pd.DataFrame(data_train.Participant.tolist()).isin(Par_test).any(1).values]

        #convert class column to 1s and 0s and further divide into X and y data sets for training and testing
        X_valid = data_valid.drop(['y','Participant'], axis=1)
        y_valid = data_valid['y'].eq('F').mul(1)

        X_valid_test = data_valid_test.drop(['y','Participant'], axis=1)
        y_valid_test = data_valid_test['y'].eq('F').mul(1)

        oversample = imblearn.over_sampling.SMOTE(random_state=42) # use smote to oversamle the minority class
        X_new, y_new = oversample.fit_resample(X_valid, y_valid)
        
        rfe = rfe.fit(X_new, y_new.values.ravel()) #fit the rfe to find the best 'i' variables
        
        logreg.fit(X_new.loc[:, rfe.support_], y_new) #fit the logreg model using those variables (columns)
        
        y_pred = logreg.predict(X_valid_test.loc[:, rfe.support_]) #predict on the left out participant
        
        tn, fp, fn, tp = confusion_matrix(y_valid_test, y_pred).ravel() #fit a confusion matrix
        test_acc = (tp + tn) / len(y_pred) #calculate accuracy on the left out test participant
        
        cv_score.append(test_acc) #store accuracy
    
    CV_all.append(np.mean(cv_score)) #calc the mean accuracy for all participants at the i # of variables
    plt.plot(CV_all)       

########################################
## Normalization of numeric variables ##
########################################
y_train = data_train['y'].eq('F').mul(1)
X_train = data_train.drop(['y','Participant'], axis=1)

y_test = data_test['y'].eq('F').mul(1)
X_test = data_test.drop(['y','Participant'], axis=1)

###############
## Fit Model ##
###############
oversample = imblearn.over_sampling.SMOTE(random_state=42)
X_new, y_new = oversample.fit_resample(X_train, y_train)

logreg = LogisticRegression(random_state=42, penalty='none')
rfe = RFE(logreg, 3)
rfe = rfe.fit(X_new, y_new.values.ravel())

logit_model=sm.Logit(y_new, sm.add_constant(X_new.loc[:, rfe.support_]))
result=logit_model.fit()
print(result.summary2())

logreg = LogisticRegression(random_state=42, penalty='none')
logreg.fit(X_new.loc[:, rfe.support_], y_new)
print(np.exp(logreg.coef_))
y_pred = logreg.predict(X_new.loc[:, rfe.support_])

## Training Performance ##
print('Training Data Performance')
print(confusion_matrix(y_new, y_pred))

print('Kappa Score (test)')
print(cohen_kappa_score(y_new, y_pred))

tn, fp, fn, tp = confusion_matrix(y_new, y_pred).ravel()
print('Negative Predicition Value (test)')
print(tn / (tn + fn))

print('Precision (test)')
print(tp / (tp + fp), '\n\n')

## Testing Performance ##
y_pred = logreg.predict(X_test.loc[:, rfe.support_])
print('Testing Data Performance')
print(confusion_matrix(y_test, y_pred))

print('Kappa Score (test)')
print(cohen_kappa_score(y_test, y_pred))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('Negative Predicition Value (test)')
print(tn / (tn + fn))

print('Precision (test)')
print(tp / (tp + fp))