import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#get training data into a dataframe
df = pd.read_csv('/Users/hernanrazo/pythonProjects/loan_prediction/train_data.csv')

#make a histogram of applicant income
#save a .png file version of the graph
histogram = plt.figure()
plt.title('Histogram of Applicant Income')
df['ApplicantIncome'].hist(bins = 50)
histogram.savefig('hist.png')

#make a box plot of applicant income
#save a .png file version of the graph
boxPlot = plt.figure()
plt.title('Box Plot for Applicant Income')
df.boxplot(column = 'ApplicantIncome')
boxPlot.savefig('boxPlot.png')

#make another box plot of applicant income but this time,
#seperate by education levels
#save a .png file version of the graph
boxPlotEd = plt.figure()
plt.title('Applicant Income by Education')
df.boxplot(column = 'ApplicantIncome', by = 'Education')
boxPlotEd.savefig('boxPlotEd.png')





















