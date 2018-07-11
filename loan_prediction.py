import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

#get training data into a dataframe
df = pd.read_csv('/Users/hernanrazo/pythonProjects/loan_prediction/train_data.csv')

#make string that holds bulk of folder path so
#I don't have to type the whole thing every time
graph_folder_path = '/Users/hernanrazo/pythonProjects/loan_prediction/graphs/'

#make a few graphs to visualize the data and
#get ideas for later manipulation 

#make a histogram of applicant income
appIncomeHist = plt.figure()
plt.title('Applicant Income Histogram')
df['ApplicantIncome'].hist(bins = 50)
appIncomeHist.savefig(graph_folder_path + 'appIncomeHist.png')

#make a box plot of applicant income
appIncomeBoxPlot = plt.figure()
plt.title('Applicant Income Box Plot')
df.boxplot(column = 'ApplicantIncome')
appIncomeBoxPlot.savefig(graph_folder_path + 'appIncomeBoxPlot.png')

#make another box plot of applicant income but this time,
#seperate by education levels
educationBoxPlot = plt.figure()
plt.title('Applicant Income by Education')
df.boxplot(column = 'ApplicantIncome', by = 'Education')
educationBoxPlot.savefig(graph_folder_path + 'educationBoxPlot.png')

#make a histogram of loan amounts
loanAmountHist = plt.figure()
plt.title('Loan Amount Value Histogram')
df['LoanAmount'].hist(bins = 50)
loanAmountHist.savefig(graph_folder_path + 'loanAmountHist.png')

#make a box plot of loan amounts
loanAmountBoxPlot = plt.figure()
plt.title('Loan Amount Value Box Plot')
df.boxplot(column = 'LoanAmount')
loanAmountBoxPlot.savefig(graph_folder_path + 'loanAmountBoxPlot.png')

#start categorical variable analysis
#make plot that shows likelihood of approval based on other catagories of 
#the applicants' information

#graph based on credit history
ch_ls_graph = plt.figure()
ch_ls_graph = pd.crosstab(df['Credit_History'], df['Loan_Status'])
ch_ls_graph.plot(kind = 'bar', stacked = True, color = ['red', 'green'], 
	grid = False, title = 'Likelihood of Approval Based on Credit History')
plt.savefig(graph_folder_path + 'ch_ls_graph.png')

#graph based on education
e_ls_graph = plt.figure()
e_ls_graph = pd.crosstab(df['Credit_History'], df['Loan_Status'])
e_ls_graph.plot(kind = 'bar', stacked = True, color = ['red', 'green'],
	grid = False, title = 'Likelihood of Approval Based on Education')
plt.savefig(graph_folder_path + 'e_ls_graph.png')

#graph based on marital status
m_ls_graph = plt.figure()
m_ls_graph = pd.crosstab(df['Credit_History'], df['Married'])
m_ls_graph.plot(kind = 'bar', stacked = True, color = ['red', 'green'],
	grid = False, title = 'Likelihood of Approval Based on Marital Status')
plt.savefig(graph_folder_path + 'm_ls_graph.png')

#now deal with empty cells and null values in the dataset

#check for empty cells and null data 
print('before: ')
print(df.apply(lambda x: sum(x.isnull()), axis = 0))

#print empty line for when reading in terminal
print(' ')

#use a frequency table to see how many self employed applicants
#got approved or denied for a loan
print(df['Self_Employed'].value_counts())

#since most self employed applicants get rejected, it is safe to 
#fill empty cells as 'no'
df['Self_Employed'].fillna('No', inplace = True)

#print empty line for when reading in terminal
print(' ')

#use a frequency table to see most common loan amount term value
print(df['Loan_Amount_Term'].value_counts())

#since most people fall into the 360 value, it is safe to 
#fill empty cells with 360
df['Loan_Amount_Term'].fillna(360, inplace = True)

#print empty line for when reading in terminal
print(' ')

#use a frequency table to see how many people do or do not have 
#a credit history that meets the loan guidelines 
print(df['Credit_History'].value_counts())

#since most applicants do have qualifying credit histories, it is 
#safe to mark empty cells with 1
df['Credit_History'].fillna(1, inplace = True)

#print empty line for when reading in terminal
print(' ')

#create pivot table for values in self employed and education categories
pivotTable = df.pivot_table(values = 'LoanAmount', index = 'Self_Employed',
	columns = 'Education', aggfunc = np.median)

def pivot_table(x):
	return pivotTable.loc[x['Self_Employed'], x['Education']]

#fill missing values
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(pivot_table,
	axis = 1), inplace = True)

#now analyze outliers and extreme values

#since the higher valued loans might actually be real and not
#necessarily outliers, just do a log transformation to cancel
#them out
LoanAmount_log = plt.figure()
plt.title('LoanAmount Log Transformation')
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins = 20)
LoanAmount_log.savefig(graph_folder_path + 'LoanAmount_log.png')

#combine applicant income and co-applicant income
#take a log transformation and graph it
combined_income = plt.figure()
plt.title('Total Income Log Transformation')
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['TotalIncome_log'].hist(bins = 20)
combined_income.savefig(graph_folder_path + 'combined_income_log.png')

#convert categorical variables into numeric ones 
variables = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
'Property_Area', 'Loan_Status']

label_encoder = LabelEncoder()

for var in variables:
	df[var] = label_encoder.fit_transform(df[var])
df.dtypes

#double check if all empty cells, null values, and categorical 
#values have been accounted for
print('after:')
print(df.isnull().sum())
print(' ')

#finally start making models 

#create function for constructing classification models and printing
#out its performance
def classification_model(model, data, predictors, outcome):

	#fit model and make prediction
	model.fit(data[predictors], data[outcome])
	prediction = model.predict(data[predictors])
	
	#configure and print accuracy
	accuracy = metrics.accuracy_score(prediction, data[outcome])
	print('Accuracy: %s' % '{0:.3%}'.format(accuracy))

	#start k-fold validation
	kf = KFold(data.shape[0], n_folds = 5, shuffle = False)
	error = []

	#filter, target, and train the model
	for train, test in kf:
		train_predictors = (data[predictors].iloc[train, :])
		train_target = data[outcome].iloc[train]
		model.fit(train_predictors, train_target)

		error.append(model.score(data[predictors].iloc[test, :],
			data[outcome].iloc[test]))

	#print cross-validation value and fit the model again
	print('Cross-Validation Score: %s' % '{0:.3%}'.format(np.mean(error)))
	model.fit(data[predictors], data[outcome])

#use the random forest algorithm
outcome_var = 'Loan_Status'
model = RandomForestClassifier(n_estimators = 100)
predictor_var = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
'Loan_Amount_Term', 'Credit_History', 'Property_Area' , 'LoanAmount_log',
'TotalIncome_log']

classification_model(model, df, predictor_var, outcome_var)

#after first training session, accuracy is extremely overfitting to the 
#training data. To fix this, figure out the most important features of 
#applicants by taking those with the highest importance matrix
series = pd.Series(model.feature_importances_, 
	index = predictor_var).sort_values(ascending = False)

print(series)
print(' ')

#retrain the model with the top five features
model = RandomForestClassifier(n_estimators = 25, min_samples_split = 25,
	max_depth = 7, max_features = 1)

predictor_var = ['Credit_History', 'TotalIncome_log', 'LoanAmount_log', 
'Dependents', 'Property_Area']

print('New model: ')
classification_model(model, df, predictor_var, outcome_var)

