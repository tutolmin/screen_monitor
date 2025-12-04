# Импортируйте библиотеку для кодирования в Base64
import base64
import requests
import json
import sys
import os

# Создайте функцию, которая кодирует файл и возвращает результат.
def encode_file(file_path):
    with open(file_path, "rb") as fid:
        file_content = fid.read()
    return base64.b64encode(file_content).decode("utf-8")

def main():
    # Проверяем, передан ли аргумент командной строки
    if len(sys.argv) < 2:
        print("Использование: python script.py <путь_к_файлу>")
        print("Пример: python script.py problems/1/slide_001.pdf")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    # Проверяем существование файла
    if not os.path.exists(file_path):
        print(f"Ошибка: файл '{file_path}' не найден")
        sys.exit(1)
    
    # Определяем MIME-тип на основе расширения файла
    if file_path.lower().endswith('.pdf'):
        mime_type = "application/pdf"
    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        mime_type = "image/png"  # или можно определить точнее по расширению
    else:
        mime_type = "application/octet-stream"  # общий тип
    
    data = {
        "mimeType": mime_type,
        "languageCodes": ["ru", "en"],
        "content": encode_file(file_path)
    }

    url = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"

    token = ""

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer {:s}".format(token),
        "x-folder-id": "b1ghg3qttqeg3e6qpgp5",
        "x-data-logging-enabled": "true"
    }

    w = requests.post(url=url, headers=headers, data=json.dumps(data))

    try:
        response_json = w.json()
#        print(response_json)
        full_text = response_json["result"]["textAnnotation"]["fullText"]
        print(full_text + "\n")
    except json.JSONDecodeError:
        print("Response Text:")
        print(w.text)
    except KeyError as e:
        print(f"Ошибка: в ответе отсутствует ожидаемое поле {e}")
        print("Полный ответ:")
        print(json.dumps(response_json, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    main()
