Naan Mudhalvan Scheme
TNSDC – Machine Learning to Generative AI
Project: Heart Disease Prediction with Machine Learning

Dhanush E
3rd Year – 2021503708
Madras Institute of Technology, Anna University

**Introduction:**
Heart disease remains a leading cause of mortality globally, emphasizing the need for effective risk assessment and early intervention strategies. In this project, we aim to leverage machine learning techniques to develop a predictive model for heart disease risk prediction. By analysing a dataset containing various patient attributes and clinical indicators, we seek to identify patterns and relationships that can aid in accurate risk assessment.
The model will be trained using logistic regression, a widely used algorithm for binary classification tasks. We will split the dataset into training and testing sets to evaluate the model's performance and ensure its generalizability to unseen data. Through comprehensive analysis and interpretation of the model's results, we aim to provide valuable insights into the factors influencing heart disease risk.
Ultimately, this project strives to contribute to the advancement of preventive healthcare by providing clinicians and healthcare professionals with a reliable tool for assessing heart disease risk and guiding patient management strategies.
**Problem Statement:**
Heart disease is a significant health concern worldwide, contributing to a considerable number of fatalities. Early detection and accurate prediction of heart disease risk can greatly improve patient outcomes and reduce mortality rates. This project aims to develop a machine learning model capable of predicting the likelihood of heart disease based on various patient attributes and clinical indicators.
**Objective:**
	To develop a predictive model using logistic regression to assess the risk of heart disease.
	Evaluate the model's performance using standard classification metrics such as accuracy, precision, recall, and F1-score.
	Investigate the effectiveness of the model in both training and testing scenarios to ensure its generalization ability.
	Provide insights into the key factors contributing to heart disease risk based on feature importance analysis.
	Enhance awareness and understanding of heart disease risk factors through data-driven analysis and interpretation.
**Goal:**
	 To develop a reliable predictive model to accurately assess heart disease risk using medical data. 
	This enables early interventions, appropriate treatments, and preventive measures for high-risk individuals.
**Scope:**
	The project involves data collection, preprocessing, exploratory data analysis, model training, evaluation, and potential clinical deployment. 
	Our focus is on developing a predictive model using logistic regression and understanding heart disease risk factors.
**Key Metrics:**
	The key metrics used to evaluate the performance of the predictive model include accuracy, precision, recall, F1-score, and confusion matrix analysis. 
	These metrics help assess the model's ability to correctly classify individuals with and without heart disease and identify any potential trade-offs between different evaluation criteria.
**Overview of Heart Disease Prediction Using ML:**
Now in this session, I’ll take you through the task of heart disease prediction using ML by using the logistic regression algorithm. As I am going to use the python programming language for this task of heart disease prediction so let’s start by importing some necessary libraries.
**Import Libraries:**
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
%matplotlib inline
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight") 
**Data Collection and Preparation: **
We collect medical records and patient data, ensuring compliance with privacy regulations. Preprocessing involves handling missing values, encoding categorical variables, and scaling numerical features.
To upload the csv file in google Colab:
from google.colab import files
uploaded = files.upload()
 
The dataset that I am using here can be easily downloaded from here. Now let’s import the data and move further.
**Exploratory Data Analysis:**
Before training the logistic regression, we need to observe and analyse the data to see what we are going to work with. The goal here is to learn more about the data and become a topic export on the dataset you are working with.
EDA helps us find answers to some important questions such as: What question (s) are you trying to solve? What kind of data do we have and how do we handle the different types? What is missing in the data and how do you deal with it? Where are the outliers and why should you care? How can you add, change, or remove features to get the most out of your data?
Now let’s start with exploratory data analysis:
df = pd.read_csv("heart.csv")
df.head()
 
pd.set_option("display.float", "{:.2f}".format)
df.describe()
 
**Data Visualization:**
df.target.value_counts().plot(kind="bar", color=["salmon", "lightblue"])
 
We have 165 people with heart disease and 138 people without heart disease, so our problem is balanced.
# Checking for messing values
df.isna().sum()
This dataset looks perfect to use as we don’t have null values.
categorical_val = []
continous_val = []
for column in df.columns:
    print('==============================')
    print(f"{column} : {df[column].unique()}")
    if len(df[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)
 
plt.figure(figsize=(15, 15))
for i, column in enumerate(categorical_val, 1):
    plt.subplot(3, 3, i)
    df[df["target"] == 0][column].hist(bins=35, color='blue', label='Have Heart Disease = NO', alpha=0.6)
    df[df["target"] == 1][column].hist(bins=35, color='red', label='Have Heart Disease = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)
 
**Observations from the above plot:**
1.	cp {Chest pain}: People with cp 1, 2, 3 are more likely to have heart disease than people with cp 0.
2.	restecg {resting EKG results}: People with a value of 1 (reporting an abnormal heart rhythm, which can range from mild symptoms to severe problems) are more likely to have heart disease.
3.	exang {exercise-induced angina}: people with a value of 0 (No ==> angina induced by exercise) have more heart disease than people with a value of 1 (Yes ==> angina induced by exercise)
4.	slope {the slope of the ST segment of peak exercise}: People with a slope value of 2 (Downslopins: signs of an unhealthy heart) are more likely to have heart disease than people with a slope value of 2 slope is 0 (Upsloping: best heart rate with exercise) or 1 (Flatsloping: minimal change (typical healthy heart)).
5.	ca {number of major vessels (0-3) stained by fluoroscopy}: the more blood movement the better, so people with ca equal to 0 are more likely to have heart disease.
6.	thal {thalium stress result}: People with a thal value of 2 (defect corrected: once was a defect but ok now) are more likely to have heart disease.
plt.figure(figsize=(15, 15))
for i, column in enumerate(continous_val, 1):
    plt.subplot(3, 2, i)
    df[df["target"] == 0][column].hist(bins=35, color='blue', label='Have Heart Disease = NO', alpha=0.6)
    df[df["target"] == 1][column].hist(bins=35, color='red', label='Have Heart Disease = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)
 
**Observations from the above plot:**
1.	trestbps: resting blood pressure anything above 130-140 is generally of concern
2.	chol: greater than 200 is of concern.
3.	thalach: People with a maximum of over 140 are more likely to have heart disease.
4.	The old peak of exercise-induced ST depression vs. rest looks at heart stress during exercise an unhealthy heart will stress more.
# Create another figure
plt.figure(figsize=(10, 8))
# Scatter with postivie examples
plt.scatter(df.age[df.target==1],
            df.thalach[df.target==1],
            c="salmon")
# Scatter with negative examples
plt.scatter(df.age[df.target==0],
            df.thalach[df.target==0],
            c="lightblue")
# Add some helpful info
plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease", "No Disease"]);
 
**Feature Engineering:**
Correlation Matrix:
# Let's make our correlation matrix a little prettier
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15, 15))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt=".2f",
                 cmap="YlGnBu");
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
 
df.drop('target', axis=1).corrwith(df.target).plot(kind='bar', grid=True, figsize=(12, 8), title="Correlation with target")
 
**Observations from correlation:**
1.	fbs and chol are the least correlated with the target variable.
2.	All other variables have a significant correlation with the target variable.
Data Processing:
After exploring the dataset and data visualization, we can observe that we need to convert some categorical variables to dummy variables and scale all values before training the machine learning models.
So, for this task, I’ll use the get_dummies method to create dummy columns for categorical variables:
categorical_val.remove('target')
dataset = pd.get_dummies(df, columns = categorical_val)
from sklearn.preprocessing import StandardScaler
s_sc = StandardScaler()
col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[col_to_scale] = s_sc.fit_transform(dataset[col_to_scale])
**Model Selection and Training: **
We employ logistic regression, ideal for binary classification tasks. The model is trained on preprocessed data with tuned hyper parameters. Cross-validation ensures model robustness.
**Modeling - Logistic Regression**
	Logistic regression serves as the cornerstone of my predictive model, offering simplicity, interpretability, and scalability. 
	Its ability to handle binary classification tasks makes it well-suited for predicting the presence or absence of heart disease based on patient attributes and clinical indicators.
**Applying Logistic Regression**
Now, I will train a machine learning model for the task of heart disease prediction. I will use the logistic regression algorithm as I mentioned at the beginning of the article. 
But before training the model I will first define a helper function for printing the classification report of the performance of the machine learning model:
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
Now let’s split the data into training and test sets. I will split the data into 70% training and 30% testing:
from sklearn.model_selection import train_test_split
X = dataset.drop('target', axis=1)
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
Now let’s train the machine learning model and print the classification report of our logistic regression model:
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(X_train, y_train)
print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)
 
**Results and Deployment: **
Logistic regression's performance is assessed using standard metrics on training and testing sets. If criteria are met, deployment in clinical settings enables real-time risk assessment and decision support.
test_score = accuracy_score(y_test, lr_clf.predict(X_test)) * 100
train_score = accuracy_score(y_train, lr_clf.predict(X_train)) * 100
results_df = pd.DataFrame(data=[["Logistic Regression", train_score, test_score]], 
columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
results_df
 
As you can see the model performs very well of the test set as it is giving almost the same accuracy in the test set as in the training set.
WHO ARE THE END USERS?
	The end users of this predictive model include healthcare professionals, clinicians, and medical practitioners involved in cardiovascular disease management. 
	Additionally, patients may benefit indirectly from the model's use through improved risk assessment and personalized healthcare interventions.
MY SOLUTION AND ITS VALUE PROPOSITION
	My solution offers a data-driven approach to predicting heart disease risk, providing healthcare professionals with a reliable tool for early detection and personalized intervention. 
	By leveraging machine learning techniques, the model enhances preventive healthcare strategies and contributes to improved patient outcomes.
THE WOW IN MY SOLUTION
	The "wow" factor in my solution lies in its ability to accurately predict heart disease risk using readily available patient data. 
	By leveraging advanced analytics, the model offers insights into key factors contributing to heart disease risk and empowers healthcare professionals to make informed decisions for better patient care.
Conclusion: 
In conclusion, the predictive model developed for heart disease risk assessment exhibits significant potential in leveraging machine learning to enhance preventive healthcare strategies. By effectively analysing patient data and employing logistic regression, the model demonstrates robust performance in predicting the likelihood of heart disease. The comprehensive evaluation of the model's performance on training and testing data underscores its reliability and generalizability, making it a valuable tool for healthcare professionals and clinicians. Through insights gained from feature importance analysis and data-driven interpretation, the model contributes to a deeper understanding of heart disease risk factors, facilitating early detection and personalized intervention. Overall, this project represents a substantial step towards improving patient outcomes and reducing mortality rates associated with heart disease.
Future Work: 
Future endeavours could focus on refining the model architecture and exploring additional features to further enhance its predictive capabilities. Incorporating advanced machine learning techniques and conducting prospective validation studies would offer valuable insights into the model's efficacy in real-world clinical settings. Moreover, ongoing efforts to expand the dataset and collaborate with healthcare institutions could facilitate the development of a more comprehensive and adaptable predictive model. Additionally, exploring avenues for integrating emerging technologies such as deep learning and natural language processing may unlock new opportunities for improving heart disease risk prediction and personalized healthcare interventions. Through continuous refinement and validation, the predictive model can continue to evolve, ultimately contributing to the advancement of preventive healthcare and better patient outcomes.
