Loan Prediction Demo
===

Description
---
A python program that predicts whether or not a loan applicant will get approved. Data includes gender, marital status, income, education level, and many other criteria. This program takes this information and makes a prediction as to whether or not the applicant is eligible for a loan. This problem is part of the [Loan Prediction Practice Problem](https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/) on the Analytics Vidhya website.  

This demo uses the scikit-learn, pandas, numpy, and matplotlib libraries. The algorithm used to make the model is the Random Forest algorithm. 

Analysis
---
First, we can make a few graphs to visualize the data and see what we might have to manipulate later on.  

Start off by making a histogram and boxplot of `applicantIncome`. 

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

```python
print(df.apply(lambda x: sum(x.isnull()), axis = 0))
```

This command prints out the total number of missing values for each column.  

To fill these values, use a frequency table to get most common value for that category. for example, a frequency table for getting data on self employed applicants that got approved or denied can be obtained by this command:

```python
print(df['Self_Employed'].value_counts())
```  

From this table, we see that most self employed applicants get rejected, so it is safe to fill missing values accordingly. To do this, use the following command:

```python
df['Self_Employed'].fillna('No', inplace = True)
```  

We can follow the same procedure above for filling empty values in `loan_amount_term` and `Credit_History`. To fill in the values in `Self_Employed` and `Education`, we create a pivot table to get median values. A function is defined to fill in the empty cells with the new values (see `loan_prediction.py` for the full code).  

To deal with the extreme values in `LoanAmount`, we will do a log transformation to get more normal distributions since the outliers could be valid:
```python
LoanAmount_log = plt.figure()
plt.title('LoanAmount Log Transformation')
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins = 20)
```  

The resulting histogram shows a more normal distribution:  

![loan amount log](https://github.com/hrazo7/loan_predictor_demo/blob/master/graphs/LoanAmount_log.png)  

For `ApplicantIncome`, we should combine them with the income of their co-applicants and apply the same log transformation:  
```python
combined_income = plt.figure()
plt.title('Total Income Log Transformation')
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['TotalIncome_log'].hist(bins = 20)
```  

The resulting histogram looks like:  
![total income log](https://github.com/hrazo7/loan_predictor_demo/blob/master/graphs/combined_income_log.png)  

Finally, we must convert all categorical variables into numeric ones using sklearn's LabelEncoder method. This assures that we won't run into problems when training our model:  

```python
variables = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
'Property_Area', 'Loan_Status']

label_encoder = LabelEncoder()

for var in variables:
	df[var] = label_encoder.fit_transform(df[var])
df.dtypes
```  

To make sure we got all values accounted for, run the following command:  

```python
print(df.isnull().sum())
```  

it should return all zeroes.  

To make the model, first we define a function that fits, trains, and prints out accuracy and cross-validation scores:


```python

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
```  

For this demo we will use the Random Forest algorithm. This algorithm is perfect for classification problems like this one because it uses the most important features to make decisions. For the dataset we are using, we can use the importance matrix it produces to select the most important features and get a better model.  

The first time we train our model, we use most of the categories in the dataset. This model returns with an accuracy that is too overfitting. To fix this, we print out the importance matrix with the following command:

```python
series = pd.Series(model.feature_importances_, 
	index = predictor_var).sort_values(ascending = False)

print(series)

```  

With this information, we can retrain the model with only the top five variables and slight modification to the other parameters:  

```python
model = RandomForestClassifier(n_estimators = 25, min_samples_split = 25,
	max_depth = 7, max_features = 1)

predictor_var = ['Credit_History', 'TotalIncome_log', 'LoanAmount_log', 
'Dependents', 'Property_Area']

print('New model: ')
classification_model(model, df, predictor_var, outcome_var)
```  

This new model has a much better score that will translate better to other data.  

Acknowledgements
---
I used [this tutorial](https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/) on the Analytics Vidhya website for guidance.  

Sources and Helpful links
---
https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/  
https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/  
https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/
https://towardsdatascience.com/the-random-forest-algorithm-d457d499ffcd