import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#get training data into a dataframe
df = pd.read_csv('/Users/hernanrazo/pythonProjects/loan_prediction/train_data.csv')

#make a few graphs to visualize the data and
#get ideas for later manipulation 

#make a histogram of applicant income
appIncomeHist = plt.figure()
plt.title('Applicant Income Histogram')
df['ApplicantIncome'].hist(bins = 50)
appIncomeHist.savefig('appIncomeHist.png')

#make a box plot of applicant income
appIncomeBoxPlot = plt.figure()
plt.title('Applicant Income Box Plot')
df.boxplot(column = 'ApplicantIncome')
appIncomeBoxPlot.savefig('appIncomeBoxPlot.png')

#make another box plot of applicant income but this time,
#seperate by education levels
educationBoxPlot = plt.figure()
plt.title('Applicant Income by Education')
df.boxplot(column = 'ApplicantIncome', by = 'Education')
educationBoxPlot.savefig('educationBoxPlot.png')


#make a histogram of loan amounts
loanAmountHist = plt.figure()
plt.title('Loan Amount Value Histogram')
df['LoanAmount'].hist(bins = 50)
loanAmountHist.savefig('loanAmountHist.png')

#make a histogram of loan amounts
loanAmountBoxPlot = plt.figure()
plt.title('Loan Amount Value Box Plot')
df.boxplot(column = 'LoanAmount')
loanAmountBoxPlot.savefig('loanAmountBoxPlot.png')

#start categorical variable analysis
#make plot that shows likelihood of approval based on other catagories of 
#the applicants' information

#graph based on credit history
ch_ls_graph = plt.figure()
ch_ls_graph = pd.crosstab(df['Credit_History'], df['Loan_Status'])
ch_ls_graph.plot(kind = 'bar', stacked = True, color = ['red', 'green'], 
	grid = False, title = 'Likelihood of Approval Based on Credit History')
plt.savefig('ch_ls_graph.png')

#graph based on education
e_ls_graph = plt.figure()
e_ls_graph = pd.crosstab(df['Credit_History'], df['Loan_Status'])
e_ls_graph.plot(kind = 'bar', stacked = True, color = ['red', 'green'],
	grid = False, title = 'Likelihood of Approval Based on Education')
plt.savefig('e_ls_graph.png')


#graph based on marital status
m_ls_graph = plt.figure()
m_ls_graph = pd.crosstab(df['Credit_History'], df['Married'])
m_ls_graph.plot(kind = 'bar', stacked = True, color = ['red', 'green'],
	grid = False, title = 'Likelihood of Approval Based on Marital Status')
plt.savefig('m_ls_graph.png')

#now deal with empty cells and null values in the dataset
#check for empty cells and null data 
print(df.apply(lambda x: sum(x.isnull()), axis = 0))

#print empty line for when reading in terminal
print(' ')

#use a frequency table to see how many self employed applicants
#got approved or denied for a loan
print(df['Self_Employed'].value_counts())

#since most self employed applicants get rejected, it is safe to 
#fill empty cells as 'no'
df['Self_Employed'].fillna('No', inplace = True)

#create pivot table for values in self employed and education categories
pivotTable = df.pivot_table(values = 'LoanAmount', index = 'Self_Employed',
	columns = 'Education', aggfunc = np.median)

def pivot_table(x):
	return pivotTable.loc[x['Self_Employed'], x['Education']]

#fill missing values
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(pivot_table,
	axis = 1), inplace = True)

#now analyze outliers and extreme values








