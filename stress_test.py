import requests
import random
import time

URL = "http://localhost:8000/v1/analyze/batch"

payload = []

for _ in range(100):   # change to 1000 later
    payload.append({
        "Speed": random.uniform(40, 140),
        "Alertness": random.uniform(0.2, 1.0),
        "Seatbelt": 1,
        "HR": random.uniform(60, 120),
        "Fatigue": random.randint(0, 10),
        "speed_change": random.uniform(0, 15),
        "prev_alertness": random.uniform(0.2, 1.0)
    })

start = time.time()

response = requests.post(URL, json=payload)

end = time.time()

print("Status:", response.status_code)
print("Time:", round(end - start, 3), "seconds")
print("Processed:", response.json().get("total_processed"))