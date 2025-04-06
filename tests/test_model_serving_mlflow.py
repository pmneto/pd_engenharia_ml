import requests
import json
import pandas as pd

url = "http://127.0.0.1:5001/invocations"


headers = {
    "Content-Type": "application/json"
}

df_input = pd.DataFrame([{
    "lat": 34.02,
    "lon": -118.25,
    "period": 2,
    "minutes_remaining": 5,
    "shot_distance": 15,
    "playoffs": 1
}])


data = {
    "inputs": df_input.to_dict(orient="records")
}


response = requests.post(url, headers=headers, data=json.dumps(data))


print("Status:", response.status_code)
print("Resposta:", response.json())
