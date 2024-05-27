# Telco-Customer-Churn-Prediction
Develop End-to-End machine learning project from EDA, model development, web interface, API, and containerization

## I. Introduction
`What is Customer Churn?`

Customer Churn is defined as when customers or subscribers discontinue doing business with a firm or service. 

Customers in the telecom industry can choose from a variety of service providers and actively switch from one to the next.

Individualized customer retention is tough because most firms have a large number of customers and can't afford to devote much time to each of them. The costs would be too great, outweighing the additional revenue. However, if a corporation could forecast which customers are likely to leave ahead of time, it could focus customer retention efforts only on these "high risk" clients.

`To reduce customer churn, telecom companies need to predict which customers are at high risk of churn`

To detect early signs of potential churn, one must first develop a holistic view of the customers and their interactions across numerous channels, including store/branch visits, product purchase histories, customer service calls, Web-based transactions, and social media interactions, to mention a few.

As a result, by addressing churn, these businesses may not only preserve their market position, but also grow and thrive.

## II. Dataset Source
Dataset:`https://www.kaggle.com/datasets/blastchar/telco-customer-churn/code?datasetId=13996&sortBy=voteCount`

## III. Methods Used
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Development
- Model Evaluation
- Web Interface Development (Streamlit)
- API Development (fastAPI)
- Containerization (Docker)

## IV. Project Structure
The project is organized as follows:
```
├── eda
│   └── eda.ipynb            # Notebook for Exploratory Data Analysis
│   └── helper_function.py   # Pyton script for data wrangling
├── model
│   └── catboost_model.cbm   # Trained CatBoost model
├── src
│   └── fast-api.py          # api script
|   └── predict.py           # model prediction script
|   └── streamlit.py         # web interface
|   └── train_model.py       # script for training model
├── Dockerfile               # Dockerfile for containerization
├── requirements.txt         # Python dependencies
├── Telco-Customer-Churn.csv # Dataset
└── README.md                # Project documentation
```

## Results
The CatBoost model achieved an `accuracy of 74%`, `Recall of 81%`, `ROC_AUC of 76%` on the test dataset. The model is deployed as a web service using FastAPI and Docker, allowing for easy integration with other systems.

## How to Run
1. Clone the repository:
   ```sh
   git clone https://github.com/beemabee/Telco-Customer-Churn-Prediction.git
   cd Telco-Customer-Churn-Prediction
   ```
2. Build and run the Docker container:
   ```sh
   docker build --no-cache -t telco-churn-prediction-app .
   docker run -it telco-churn-prediction-app
   ```
   
## Conclusion
This project demonstrates an end-to-end machine learning workflow, from data analysis to model deployment. The use of Docker ensures that the application can be easily deployed and scaled.
