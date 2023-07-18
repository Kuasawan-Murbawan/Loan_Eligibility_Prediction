Website Link -> https://kuasawan-murbawan-loan-eligibility-prediction-index-cgzl9m.streamlit.app/
# Loan-Eligibility-Prediction
<b>Contents<b/>
1. [Introduction](##Introduction)
2. [Dataset](##dataset)
3. [Algo used](##algorithm-used)
4. [Streamlit](##publishing-to-the-web)



## Introduction

In finance, one of the determinants of an individual’s financial well being is their ability to access the credit. Loans are an important tool that can be used by an individual to pay their necessities. However, not everyone may be eligible for a loan. We aim to study the loan data to determine the factors that can affect loan eligibility.

To achieve the aim of our study, we undertook several approaches such as exploratory data analysis (EDA), feature selection, and modeling. 

The integration of the findings into a web application will allow the user to input information about loan applicants and receive a prediction about their eligibility. This will enable users to make more informed decisions about loan approvals.

## Dataset
The "Loan_Data.csv" dataset contains information about borrowers and loans granted by a financial institution. The dataset can be used to build a machine learning model that can predict loan approval based on the borrower's characteristics and loan details. We import the dataset from kaggle if you want to access it. 
([Eligibility Prediction for Loan | Kaggle](https://www.kaggle.com/datasets/devzohaib/eligibility-prediction-for-loan))

## Algorithm Used
We used 5 techniques as our feature selection methods, Recursive Feature Elimination (RFE), Correlation-based Feature Selection (CFS), Principal Component Analysis (PCA), Mutual Information-based Feature Selection (MIFS), Univariate Feature Selection (UFS) and choose **accuracy** as our performance analysis to make it simple to understand.
| Feature Selection Techniques| Accuracy|  
| ----------- | ----------- |  
|UFS (5 Features |**0.834** |
|MIFS and 5 features|**0.834**|
|PCA and 5 components|**0.715278**|
|SelectKBest (CFS) and 6 features|**0.834**|
|RFE and 6 features|**0.8334**|

([Loan Prediction Feature Selection Code | Google Collab](https://colab.research.google.com/drive/1hqPVKwBrHwxPriaVLfFVZAR-uwiKuq6z?usp=sharing))

## Publishing to the web
([Loan-Eligibility-Predictor | Streamlit](https://kuasawan-murbawan-loan-eligibility-prediction-index-cgzl9m.streamlit.app/))

In this project, we have developed a machine learning model to predict certain outcomes based on user input data. The model was trained using various feature selection techniques but we only use 3 model to demo because the accuracy for most of the model is the same. We include no feature selection, Recursive Feature Elimination (RFE), and Principal Component Analysis (PCA). We then exported the models into pickle files using joblib.
Interface:

To provide a simple user interface for making predictions, we created a Python and Streamlit-based web application. The interface prompts the user to input all the necessary columns based on their data. Based on the feature selection technique chosen, only certain columns will be sent to the model for prediction.

![](https://lh4.googleusercontent.com/duS4otegiMaMi_m8GYIcpg5qtF7EYAB08h1RofR6qv8dwtvd1tk5OIeFKS5gNjUhGGGdNXcf85T9m48Xi8V8Sbc3Qoj67Wi_mrD3CikQWeEueI95skW17LfJqioPeYli_nWizqByeGcozO_arc8RI-8)

The user interface we have created is very simple and intuitive. It prompts the user to input all the necessary columns based on their data. Once the user inputs the data, we repack it inside a dataframe to ensure the order of the columns matches that of the trained model.

  

Based on the feature selection technique chosen, only certain columns will be sent to the model for prediction. For example, the regular model will send all the columns to the model, while the RFE model will only need six columns to make a prediction.

  

After we get the user input, we import the model and predict the user's data using the model we just loaded. Then we display the output to the user.

![](https://lh4.googleusercontent.com/13fLU2YlA5EWUMTDbzmJHPV-6qKuFihiKQUA4OtThOxpcS8h2_xl5SzCOem5LtoadOxAT7YOkM_2UivJojsarIzSRc274zYumAO3dQDK4NyBn0RIvA7hEfvzY_QPUsUncsbZGBITyzqGNErqUMyIPxo)
 
After all the models were trained, we exported them into pickle files using joblib. We exported two pickle files, one for the regular model called 'reg_clf.pkl', and the other for the RFE model called 'rfe_clf.pkl'.
```  
# function if the user choose RFE model  
def  rfe_display(option):  
  
rfe_user_input = pd.DataFrame([[Gender, Married, Education, Self_Employed, Credit_History, Property_Area]])  
  
rfe_clf_model = joblib.load("rfe_clf.pkl")  
prediction = rfe_clf_model.predict(rfe_user_input)
```
