Loan Prediction Demo
===

Description
---
A python program that predicts whether or not a loan applicant will get approved. Data includes gender, marital status, income, education level, and many other criteria. This program takes this information and makes a prediction as to whether or not the applicant is eligible for a loan. This problem is part of the [Loan Prediction Practice Problem](https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/) at the Analytics Vidhya website.  

Analysis
---
First, we can make a few graphs to visualize the data and see what we might have to manipulate later on.  

First make a histogram and boxplot of `applicantIncome`. 

![applicant Income histogram](https://github.com/hrazo7/loan_predictor_demo/blob/master/graphs/appIncomeHist.png) ![applicant income boxplot](https://github.com/hrazo7/loan_predictor_demo/blob/master/graphs/appIncomeBoxPlot.png)  

We can see there are a few extreme values. These values can still be valid due the applicants having high income variations. So to get a better picture, split applicant income by education levels.  

![applicant income boxplot by education](https://github.com/hrazo7/loan_predictor_demo/blob/master/graphs/educationBoxPlot.png)  

Using this boxplot, it is clearly visible that the range in data is higher for those with higher education. There is also no significant difference in mean income between those that graduated and those that did not.  

Now we can make a histogram and boxplot of loan amount values.  

![loan amount histogram](https://github.com/hrazo7/loan_predictor_demo/blob/master/graphs/loanAmountHist.png) ![loan amount boxplot](https://github.com/hrazo7/loan_predictor_demo/blob/master/graphs/loanAmountBoxPlot.png)  

There are still some visible values that could be outliers. This might be normal, but further analysis is still necessary.  

Now we focus on categorical variables and how they influence an applicants' loan approval. To do this, make bar charts to visualize the likelihood of getting a loan based on an applicants specific situation. The categorical variables graphed here are 
`Education`, `Married`, and `Credit_History`.  

![Education Level](https://github.com/hrazo7/loan_predictor_demo/blob/master/graphs/e_ls_graph.png) ![Marital Status](https://github.com/hrazo7/loan_predictor_demo/blob/master/graphs/m_ls_graph.png) ![Credit History](https://github.com/hrazo7/loan_predictor_demo/blob/master/graphs/ch_ls_graph.png)  

There is a clear preference for people that are educated, married, and with existing credit history. But to get the best possible picture, we need to clean the datatset by fixing null and empty values. To do this, we use the below command:  

`print(df.apply(lambda x: sum(x.isnull()), axis = 0))`  

This command prints out the total number of missing values for each column.  

To fill these values, use a frequency table to get most common value for that category. for example, a frequency table for getting data on self employed applicants that got approved or denied can be obtained by this command:  

`print(df['Self_Employed'].value_counts())`  

From this table, we see that most self employed applicants get rejected, so it is safe to fill missing values accordingly. To do this, use the following command:  

`df['Self_Employed'].fillna('No', inplace = True)`  

We can follow the same procedure above for filling empty values in `loan_amount_term` and `Credit_History`. To fill in the values in `Self_Employed` and `Education`, we create a pivot table to get median values. A function is defined to fill in the empty cells with the new values (see `loan_prediction.py` for the full code).  













Sources and Helpful links
---
https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/  
https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/  
https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/
https://towardsdatascience.com/the-random-forest-algorithm-d457d499ffcd