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
#make plot that shows likelihood of approval based on various catagories of 
#the applicants

#graph based on credit history
ch_ls_graph = plt.figure()
ch_ls_graph = pd.crosstab(df['Credit_History'], df['Loan_Status'])
ch_ls_graph.plot(kind = 'bar', stacked = True, color = ['red', 'green'], 
	grid = False, title = 'Likelihood of Approval Based on Credit History')
plt.savefig('ch_ls_graph.png')

























