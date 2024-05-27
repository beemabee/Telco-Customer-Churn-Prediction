import pandas as pd 
from catboost import CatBoostClassifier

# load the trained model
MODEL_PATH = '/app/model/catboost_model.cbm'
model = CatBoostClassifier()
model.load_model(MODEL_PATH)

def predict_churn(user_input):
    try:
        # prepare data for prediction
        user_data = pd.DataFrame([user_input])

        # make prediction
        prediction = model.predict_proba(user_data)[:,1][0]

        return {'Churn probability' : float(prediction)}
    
    except Exception as e:
        return {'error' : str(e)}
    

if __name__ == '__main__':
    # Ask user for input
    customerID = '6464-UIAEA'
    print('Please enter the following information:')
    gender = input('Gender (Male/Female): ').strip()
    senior_citizen = int(input('Senior Citizen (0/1): ').strip())
    partner = input('Partner (Yes/No): ').strip()
    dependents = input('Dependents (Yes/No): ').strip()
    tenure = int(input('Tenure (months): ').strip())
    phone_service = input('Phone Service (Yes/No): ').strip()
    multiple_lines = input('Multiple Lines (Yes/No): ').strip()
    internet_service = input('Internet Service (DSL/Fiber optic/No): ').strip()
    online_security = input('Online Security (Yes/No): ').strip()
    online_backup = input('Online Backup (Yes/No): ').strip()
    device_protection = input('Device Protection (Yes/No): ').strip()
    tech_support = input('Tech Support (Yes/No): ').strip()
    streaming_tv = input('Streaming TV (Yes/No): ').strip()
    streaming_movies = input('Streaming Movies (Yes/No): ').strip()
    contract = input('Contract (Month-to-month/One year/Two year): ').strip()
    paperless_billing = input('Paperless Billing (Yes/No): ').strip()
    payment_method = input('Payment Method (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic)): ').strip()
    monthly_charges = float(input('Monthly Charges ($): ').strip())
    total_charges = float(input('Total Charges ($): ').strip())

    new_customer_data = pd.DataFrame({
        'customerID': [customerID],
                'gender': [gender],
                'SeniorCitizen': [senior_citizen],
                'Partner': [partner],
                'Dependents': [dependents],
                'tenure': [tenure],
                'PhoneService': [phone_service],
                'MultipleLines': [multiple_lines],
                'InternetService': [internet_service],
                'OnlineSecurity': [online_security],
                'OnlineBackup': [online_backup],
                'DeviceProtection': [device_protection],
                'TechSupport': [tech_support],
                'StreamingTV': [streaming_tv],
                'StreamingMovies': [streaming_movies],
                'Contract': [contract],
                'PaperlessBilling': [paperless_billing],
                'PaymentMethod': [payment_method],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges]
    })

    # predict churn probability using the model
    churn_prob = model.predict_proba(new_customer_data)[: , 1]

    # format churn probability
    formatted_churn_prob = '{:.2%}'.format(churn_prob.item())

    # print result
    print(f"Churn Probability: {formatted_churn_prob}")