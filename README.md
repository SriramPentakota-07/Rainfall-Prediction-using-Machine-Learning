## Rainfall Prediction using Machine Learning

### Overview
This project predicts whether it will rain tomorrow using historical rainfall data of India. It uses a machine learning model trained on past data to make predictions.

### Key Features
- Predicts rainfall as **Yes or No**
- Uses historical rainfall dataset from India
- Supports region-based prediction (Andhra Pradesh - Coastal Andhra)
- Automatically takes today’s date and predicts for tomorrow
- Simple and easy-to-understand implementation

### Libraries Used
- pandas
- scikit-learn
- kagglehub
- datetime
- os

### How it Works
- Loads dataset from Kaggle
- Cleans and preprocesses the data
- Filters data for Andhra Pradesh region
- Creates a target column (Rain: Yes/No)
- Trains a Decision Tree model
- Uses average rainfall values as input
- Predicts whether it will rain tomorrow

### Output
- Displays today’s date
- Displays tomorrow’s date
- Shows rain prediction (Yes/No)

### Note
This prediction is based on historical data patterns and not real-time weather conditions.
