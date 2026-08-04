import pandas as pd
import kagglehub
import os
from datetime import datetime, timedelta
from sklearn.tree import DecisionTreeClassifier

path = kagglehub.dataset_download("rajanand/rainfall-in-india")
file_path = os.path.join(path, "rainfall in india 1901-2015.csv")

data = pd.read_csv(file_path)

data.columns = data.columns.str.strip()
data.fillna(0, inplace=True)

print(list(set(data['SUBDIVISION'])))
location = input().lower()
data = data[data['SUBDIVISION'].str.lower() == location]

if data.empty:
    print(f"Error: No data found for the given location '{location}'. Please check the spelling or available subdivisions.")
else:
    today = datetime.today().strftime("%Y-%m-%d")
    tomorrow = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Location: {location}")
    print("Today's Date:", today)
    print("Tomorrow's Date:", tomorrow)

    data['Rain'] = data['ANNUAL'].apply(lambda x: 1 if x > 1000 else 0)

    features = ['JAN','FEB','MAR','APR','MAY','JUN',
                'JUL','AUG','SEP','OCT','NOV','DEC']

    X = data[features]
    y = data['Rain']

    model = DecisionTreeClassifier()
    model.fit(X, y)

    avg_values_dict = data[features].mean().to_dict()
    prediction_df = pd.DataFrame([avg_values_dict])

    prediction = model.predict(prediction_df)

    if prediction[0] == 1:
        print(f"Rain Prediction for {location} on {tomorrow}: YES")
    else:
        print(f"Rain Prediction for {location} on {tomorrow}: NO")
