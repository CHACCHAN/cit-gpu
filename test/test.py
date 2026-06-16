import requests
import base64

url = "https://garbage-caviar-expert.ngrok-free.dev/v1/images/generations"
data = {
    "prompt": "カレーライス",
    "size": "1024x1024",
    "steps": 50
}

response = requests.post(url, json=data, timeout=None)

if response.status_code == 200:
    b64_data = response.json()["data"][0]["b64_json"]
    with open("output.png", "wb") as f:
        f.write(base64.b64decode(b64_data))
    print("save as output")
else:
    print(f"error: {response.status_code}")