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

location = "COASTAL ANDHRA PRADESH" # Corrected location name
data = data[data['SUBDIVISION'] == location]

if data.empty:
    print(f"Error: No data found for the given location '{location}'. Please check the spelling or available subdivisions.")
else:
    today = datetime.today().strftime("%Y-%m-%d")
    tomorrow = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    print("Location: Andhra Pradesh (Coastal Andhra)")
    print("Today's Date:", today)
    print("Tomorrow's Date:", tomorrow)

    data['Rain'] = data['ANNUAL'].apply(lambda x: 1 if x > 1000 else 0)

    features = ['JAN','FEB','MAR','APR','MAY','JUN',
                'JUL','AUG','SEP','OCT','NOV','DEC']

    X = data[features]
    y = data['Rain']

    model = DecisionTreeClassifier()
    model.fit(X, y)

    avg_values = data[features].mean().values.reshape(1, -1)

    prediction = model.predict(avg_values)

    if prediction[0] == 1:
        print("Rain Prediction for Andhra Pradesh on", tomorrow, ": YES")
    else:
        print("Rain Prediction for Andhra Pradesh on", tomorrow, ": NO")

==== Sample Output ====
Location: Andhra Pradesh (Coastal Andhra)
Today's Date: 2026-04-06
Tomorrow's Date: 2026-04-07
Rain Prediction for Andhra Pradesh on 2026-04-07 : YES
