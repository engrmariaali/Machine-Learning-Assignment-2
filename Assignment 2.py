# Maria Ali
# Machine Learning Assignment 2
# Using Simple Linear Regression to predict the brain weight(grams) from the head size(cm^3)

# Importing the libraries and the dataset for evaluation
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('dataset.csv')
# Assembling the values for "p" and "q":
p = dataset.iloc[:, 2:3].values
q = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
p_train, p_test, q_train, q_test = train_test_split(p, q, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(p_train, q_train)

# Predicting the Test set results
q_pred = regressor.predict(p_test)

# Visualising the Training set results
plt.scatter(p_train, q_train, color = 'orange') # Creating Scatter Plot
plt.plot(p_train, regressor.predict(p_train), color = 'brown', label='Best Fit Line') # Creating the Best Fit Line
tnrfont = {'fontname':'Times New Roman'} # Setting the font "Times New Roman"
plt.title('Brain Weight (grams) vs Head Size (cm^3) - Training Set',**tnrfont) # Setting the Title
plt.xlabel('Head Size (cm^3)', **tnrfont) # Labelling x-axis
plt.ylabel('Brain Weight (grams)', **tnrfont) # Labelling y-axis
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5) # Creating Grid
leg = plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1); # Creating Legend
plt.show()

# Visualising the Test set results
plt.scatter(p_test, q_test, color = 'orange') # Creating Scatter Plot
plt.plot(p_train, regressor.predict(p_train), color = 'brown', label='Best Fit Line') # Creating the Best Fit Line
tnrfont = {'fontname':'Times New Roman'} # Setting the font "Times New Roman"
plt.title('Brain Weight (grams) vs Head Size (cm^3)- Test Set',**tnrfont) # Setting the Title
plt.xlabel('Head Size (cm^3)', **tnrfont) # Labelling x-axis
plt.ylabel('Brain Weight (grams)', **tnrfont) # Labelling y-axis
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5) # Creating Grid
leg = plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1); # Creating Legend
plt.show()
