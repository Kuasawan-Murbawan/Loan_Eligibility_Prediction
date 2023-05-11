import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.title("Loan Eligibility Predictor")

# Function if the user choose no feature selection
def non_display(option):
    st.write("Baseline accuracy with all features is 0.7916666666666666")
    reg_df_pred = pd.DataFrame([[Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area]])
    reg_cld_model = joblib.load("reg_clf.pkl")
    prediction = reg_cld_model.predict(reg_df_pred)

    if st.button("Predict"):
        if prediction == 1:
            st.write("You are eligible")
        else:
            st.write("You are not eligible")
   
# function if the user choose RFE model
def rfe_display(option):
    st.write("Accuracy with RFE and 6 features is 0.8333333333333334")
    st.write("The selected features are Gender, Married, Education, Self_Employed, Credit_History, Property_Area")
    rfe_df_pred = pd.DataFrame([[Gender, Married, Education, Self_Employed, Credit_History, Property_Area]])
    #columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
    rfe_clf_model = joblib.load("rfe_clf.pkl")
    prediction = rfe_clf_model.predict(rfe_df_pred)

    if st.button("Predict"):
        if prediction == 1:
            st.write("You are eligible")
        else:
            st.write("You are not eligible")

            
 # option for PCA model
def pca_display(option):

    # Import necessary libraries
    import pandas as pd    
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeClassifier

    # Load loan dataset from a CSV file
    loan_df = pd.read_csv(r"Loan_Data_df.csv")

    # Split the dataset into features (X) and target (y)
    y = loan_df['Loan_Status']
    X = loan_df.drop(['Loan_Status'], axis=1)

    # Apply PCA to reduce the dimensionality of the feature space
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X)

    # Get the names of selected columns
    col_names = [f"PC{i+1}" for i in range(pca.n_components_)]

    # Train a Ridge Classifier model
    pca_clf = RidgeClassifier()
    pca_clf.fit(X_pca, y)

    # Create a pandas DataFrame with user input data
    user_data = pd.DataFrame([[Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area,0]], columns=X.columns)

    # Apply PCA to user input data
    user_data_pca = pca.transform(user_data)

    # Make a prediction using the trained model
    prediction = pca_clf.predict(user_data_pca)

    # Split the PCA-transformed features into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=109)

    # Predict the response for test dataset
    y_pred = pca_clf.predict(X_test)

    # Compute the accuracy score of the model
    accuracy = metrics.accuracy_score(y_test, y_pred)

    # Print the accuracy score and the names of the selected columns
    st.write(f'Accuracy with PCA and {pca.n_components_} components is {accuracy}')
    st.write(f'Selected columns: {col_names}')


    # Print the prediction
    if st.button("Predict"):
        if prediction == 1:
            st.write("You are eligible")
        else:
            st.write("You are not eligible")

    from PIL import Image
    
    
    image = Image.open(r"heatmap.png")
    
    st.image(image, caption=' The values in the heatmap indicate the contribution of each original feature to each principal component. The values closer to 1 or -1 indicate a stronger contribution to the principal component, while values closer to 0 indicate a weaker contribution')




st.divider()

# User inputs

col1, col2, col3 = st.columns(3)

with col1:
        #st.write("Column 1")
        Gender = st.radio("Gender (0-Male, 1-Female)", (0, 1))
        Education = st.radio("Education (0-Not Graduate, 1-Graduate)",(0, 1))
        Married = st.radio("Married (0-No, 1-Yes)", (0, 1))
        Self_Employed = st.radio("Self Employed? (0-No, 1-Yes)", (0, 1))
        

with col2:
        #st.write("Column 2")
        Dependents = st.selectbox("Dependents", (1, 2, 3))
        LoanAmount = st.slider("Loan Amount x1000",9,600)
        Loan_Amount_Term = st.slider("Loan Term (in months)", 36,480)



with col3:
        #st.write("Column 3")
        ApplicantIncome = st.slider("Applicant Income",150,81000) 
        CoapplicantIncome = st.slider("Coapplicant Income",0,40000)
        # TODO: check min & max applicant income
        Credit_History = st.radio("Credit History", (0,1))
        Property_Area = st.radio("Property Area (0-Rural, 1-Semiurban, 2-Urban)", (0,1,2))

st.divider()

option = st.selectbox("Which feature selection methods do you want to use?", ("No feature selection","Recursive Feature Elimination (RFE)","Principal Component Analysis (PCA)"))

#if st.button("Predict"):
#        st.write("You are not eligible")


if option=="No feature selection":
    non_display(option)
elif option=="Recursive Feature Elimination (RFE)":
    rfe_display(option)
elif option=="Principal Component Analysis (PCA)":
    pca_display(option)
