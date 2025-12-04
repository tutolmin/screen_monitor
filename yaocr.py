# Импортируйте библиотеку для кодирования в Base64
import base64
import requests
import json

# Создайте функцию, которая кодирует файл и возвращает результат.
def encode_file(file_path):
  with open(file_path, "rb") as fid:
    file_content = fid.read()
  return base64.b64encode(file_content).decode("utf-8")

#data = {"mimeType": "image/png",
data = {"mimeType": "application/pdf",
        "languageCodes": ["ru","en"],
        "content": encode_file("problems/1/slide_001.pdf")}
#        "content": encode_file("images/Screenshot 2025-10-23 09-58-38.png")}
#        "content": encode_file("/var/tmp/screens/original_20251021_113151_1.png")}
#        "content": encode_file("images/8930.jpg")}

url = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"

token = ""

headers= {"Content-Type": "application/json",
          "Authorization": "Bearer {:s}".format(token),
          "x-folder-id": "b1ghg3qttqeg3e6qpgp5",
          "x-data-logging-enabled": "true"}

w = requests.post(url=url, headers=headers, data=json.dumps(data))

#print(f"Status Code: {w.status_code}")

try:
    response_json = w.json()
    full_text = response_json["result"]["textAnnotation"]["fullText"]
#    print("Извлеченный текст:")
    print(full_text + "\n")
#    print("Response JSON:")
#    print(json.dumps(response_json, indent=2, ensure_ascii=False))
except json.JSONDecodeError:
    print("Response Text:")
    print(w.text)
