import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Churn_Modelling.csv")

st.title("Bank Customer Churn Model")

st.sidebar.subheader("Geography")

geography_options = ('France', 'Spain', 'Germany')
geography = st.sidebar.radio(label= "Choose your Geography", options= geography_options)

credit_score_min_value = 300
credit_score_max_value = 850
credit_score = st.slider("Enter customer's Credit Score", int(credit_score_min_value), int(credit_score_max_value), int(credit_score_min_value), 1)


genders = ("Male", "Female")
gender = st.selectbox("Select customer's Gender", options=genders)

age = st.number_input("Enter customer's Age", min_value=18, max_value = 150)

tenure_min_value = 0
tenure_max_value = 10
tenure = st.slider("Enter customer's Tenure (Number of Years)", int(tenure_min_value), int(tenure_max_value), int(tenure_min_value), 1)

balance_min_value = 0
balance_max_value = 250000
balance = st.slider("Enter customer's Balance", int(balance_min_value), int(balance_max_value), int(balance_min_value), 1)


num_of_prods_opts = (1,2,3,4)
num_of_products = st.selectbox("Enter customer's Number of Products", options=num_of_prods_opts)

has_cr_card = st.number_input("Enter binary flag for whether customer holds a Credit Card", min_value=0, max_value=1)

is_active_member = st.number_input("Enter binary flag for whether customer active", min_value=0, max_value=1)

estimated_salary_min_value = 0
estimated_salary_max_value = 200000
estimated_salary = st.slider("Enter customer's Estimated Salary", int(estimated_salary_min_value), int(estimated_salary_max_value), int(estimated_salary_min_value), 1)


def get_transformed_data(df):
    df.drop(columns=["RowNumber","CustomerId","Surname"],inplace=True)
    df.rename(columns={"Exited":"Churned"},inplace=True)
    df["Account_Balance"] = df["Balance"].apply(lambda x: "Zero Balance" if x==0 else "More Than zero Balance")
    df.drop(columns="Balance",inplace=True)
    cat_cols = ['Geography', 'Gender', 'NumOfProducts', 'Account_Balance']
    df = pd.get_dummies(columns=cat_cols, data=df)
    df["Age"] = np.log(df["Age"])
    X = df.drop(columns=["Churned"])
    return X



def make_prediction(geography, credit_score, gender, age, tenure, balance, num_of_products,\
                    has_cr_card, is_active_member, estimated_salary):
    
    log_age = np.log(age)
    print(f"age: {age} and log_age:{log_age}")
    dict = {"CreditScore": credit_score, "Age":age, "Tenure":tenure, "HasCrCard":has_cr_card, 
            "IsActiveMember":is_active_member, "EstimatedSalary":estimated_salary}
    
    to_binary_geo = lambda x: 1 if geography == x else 0
    dict["Geography_France"] = to_binary_geo("France")
    dict["Geography_Spain"] = to_binary_geo("Spain")
    dict["Geography_Germany"] = to_binary_geo("Germany")

    to_binary_gender = lambda x : 1 if gender == x else 0
    dict['Gender_Female'] = to_binary_gender("Female")
    dict['Gender_Male'] = to_binary_gender("Male")

    to_binary_number_of_products = lambda x : 1 if num_of_products == x else 0
    dict['NumOfProducts_1'] = to_binary_number_of_products(1)
    dict['NumOfProducts_2'] = to_binary_number_of_products(2)
    dict['NumOfProducts_3'] = to_binary_number_of_products(3)
    dict['NumOfProducts_4'] = to_binary_number_of_products(4)
    
    balance_type_func = lambda x: 'More Than Zero Balance' if int(x) != 0 else "Zero Balance" 
    balance_type = balance_type_func(balance)
    binary_more_than_zero_balance_type = lambda x: 1 if x == 'More Than Zero Balance' else 0
    binary_zero_balance_type = lambda x: 1 if x == 'Zero Balance' else 0
    dict["Account_Balance_More Than zero Balance"] = binary_more_than_zero_balance_type(balance_type)
    dict["Account_Balance_Zero Balance"] = binary_zero_balance_type(balance_type)

    features = ['CreditScore', 'Age', 'Tenure', 'HasCrCard', 'IsActiveMember',
       'EstimatedSalary', 'Geography_France', 'Geography_Germany',
       'Geography_Spain', 'Gender_Female', 'Gender_Male', 'NumOfProducts_1',
       'NumOfProducts_2', 'NumOfProducts_3', 'NumOfProducts_4',
       'Account_Balance_More Than zero Balance',
       'Account_Balance_Zero Balance']
    
    test_vector = []
    for feature in features:
        test_vector.append(dict[feature])
    
    X = get_transformed_data(df)

    print(f"test_vector before scaling : {test_vector}")
    scaler = StandardScaler()
    scaler.fit_transform(X)
    test_vector = scaler.transform(np.array([test_vector]))
    print(f"test_vector after scaling: {test_vector}")

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    result = model.predict(test_vector)[0]
    print(f"result: {result}")
    proba_res = model.predict_proba(test_vector)[:][:,1]
    print(f"proba_res: {proba_res}")
    return result, proba_res

result, proba_res = make_prediction(geography, credit_score, gender, age, tenure, balance, num_of_products,\
                        has_cr_card, is_active_member, estimated_salary)

proba_res = round(proba_res[0]*100,2)
if st.button("Predict"):
    if result == 1:
        st.warning(f"WARNING! High probability of customer loss predicted ( %{proba_res} ), prompt action recommended.")
    else:
        st.success(f"HAPPY NEWS! The result indicates a low churn risk ( %{proba_res} ) and high loyalty likelihood.")