import uvicorn
import pandas as pd 
from fastapi import FastAPI 
from catboost import CatBoostClassifier

# path to the model
MODEL_PATH = r'D:\DS_ML_Project\Telco-Customer-Churn-Prediction\Model\catboost_model.cbm'

# function to load the trained model
def load_model():
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model 

# function to predict churn probability from data in Dataframe format
def get_churn_prob(data, model):

    # convert incoming data into a Dataframe
    dataframe = pd.DataFrame.from_dict(data, orient='index').T 

    # make the prediction
    churn_prob = model.predict_proba(dataframe)[0][1]

    return churn_prob

# load the model
model = load_model()


# create the fastAPI application
app = FastAPI(title='Telco Customer Churn API', version='1.0.0')
@app.get('/')

def index():
    return {'message': 'CHURN Prediction API'}

# define API endpoint
@app.post('/predict/')

def predict_churn(data: dict):

    # get the prediction
    churn_prob = get_churn_prob(data, model)

    # return the prediction
    return {'Churn Probability' : churn_prob}

# run the application
if __name__ == '__main__':
    uvicorn.run('fast-api:app', host='127.0.0.1', port=5000)