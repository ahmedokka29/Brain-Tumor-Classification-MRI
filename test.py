import requests

url = 'http://localhost:9696/predict'


data = {'url': 'https://raw.githubusercontent.com/ahmedokka29/ml-zoomcamp/main/11_capstone_1/data/Testing/glioma_tumor/image(1).jpg'}

result = requests.post(url, json=data).json()
print(result)