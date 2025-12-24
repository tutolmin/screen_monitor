#!/usr/bin/env python
"""
Скрипт для обработки архива фотографий.
Принимает путь к каталогу с изображениями как аргумент командной строки.
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path
from PIL import Image, ImageEnhance
# Пример OCR библиотеки - tesseract через pytesseract
# pip install pytesseract
# Также нужно установить tesseract.exe
# import pytesseract
import yandexcloud
from yandex.cloud.iam.v1.iam_token_service_pb2 import (CreateIamTokenRequest)
from yandex.cloud.iam.v1.iam_token_service_pb2_grpc import IamTokenServiceStub
import time
from datetime import datetime, timedelta
import jwt                                                                                                                          
import json
import threading
from dotenv import load_dotenv
#import cv2
#import numpy as np
import base64
import requests

# Загружаем переменные окружения из файла .env
load_dotenv()

class YandexCloudAuthManager:
    """Менеджер аутентификации Yandex Cloud с автоматическим обновлением токенов"""

    def __init__(self, service_account_key_path):
        """
        Инициализация менеджера аутентификации

        Args:
            service_account_key_path: путь к JSON-файлу с ключом сервисного аккаунта
        """
        self.service_account_key_path = service_account_key_path
        self.iam_token = None
        self.token_expires_at = None
        self.lock = threading.RLock()  # Для потокобезопасного доступа
        self._load_service_account_key()

    def log_message(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {message}")

    def _load_service_account_key(self):
        """Загрузка ключа сервисного аккаунта"""
        with open(self.service_account_key_path, 'r') as f:
            key_data = json.load(f)
            self.service_account_id = key_data['service_account_id']
            self.key_id = key_data['id']
            self.private_key = key_data['private_key']

    def _create_jwt(self):
        """Создание JWT-токена для получения IAM-токена"""
        now = int(time.time())
        payload = {
            'aud': 'https://iam.api.cloud.yandex.net/iam/v1/tokens',
            'iss': self.service_account_id,
            'iat': now,
            'exp': now + 3600  # JWT действует 1 час
        }

        return jwt.encode(
            payload,
            self.private_key,
            algorithm='PS256',
            headers={'kid': self.key_id}
        )

    def _get_new_iam_token(self):
        """Получение нового IAM-токена от Yandex Cloud API"""
        try:
            # Создаем JWT
            jwt_token = self._create_jwt()

            # Инициализируем SDK с ключом сервисного аккаунта
            sdk = yandexcloud.SDK(service_account_key={
                "service_account_id": self.service_account_id,
                "id": self.key_id,
                "private_key": self.private_key
            })

            # Получаем IAM-токен
            iam_service = sdk.client(IamTokenServiceStub)
            response = iam_service.Create(CreateIamTokenRequest(jwt=jwt_token))

            # Токен действителен 12 часов, но обновляем через 11 для надежности
            self.iam_token = response.iam_token
            self.token_expires_at = datetime.now() + timedelta(hours=11)

            self.log_message(f"Получен новый IAM-токен, действителен до: {self.token_expires_at}")
            return self.iam_token

        except Exception as e:
            self.log_message(f"Ошибка получения IAM-токена: {str(e)}")
            raise

    def get_valid_token(self):
        """
        Получение действительного IAM-токена.
        Если токен отсутствует или истек срок действия - обновляет его.

        Returns:
            Действительный IAM-токен
        """
        with self.lock:
            # Если токена нет или срок истек (или истекает через 5 минут)
            if (self.iam_token is None or
                self.token_expires_at is None or
                datetime.now() >= self.token_expires_at - timedelta(minutes=5)):

                self.log_message("Токен отсутствует или скоро истечет, обновляем...")
                return self._get_new_iam_token()

            # Токен действителен
            time_remaining = self.token_expires_at - datetime.now()
            self.log_message(f"Используется существующий токен, осталось: {time_remaining}")
            return self.iam_token

    def force_refresh(self):
        """Принудительное обновление токена"""
        with self.lock:
            self.log_message("Принудительное обновление токена...")
            return self._get_new_iam_token()

def parse_arguments():
    """Чтение имени каталога через аргумент командной строки."""
    parser = argparse.ArgumentParser(
        description="Обработать архив фотографий в указанной папке."
    )
    parser.add_argument(
        "input_directory",
        type=str,
        help="Путь к каталогу с фотографиями для обработки."
    )
    parser.add_argument(
        "-o", "--output_directory",
        type=str,
        default=None,
        help="Путь к каталогу для сохранения результатов (по умолчанию: input_dir/processed)."
    )
    return parser.parse_args()


def get_photo_files(directory, extensions=None):
    """
    Получить список файлов фото по типу или маске.
    """
    input_dir = directory.resolve()
    if not input_dir.is_relative_to(Path.cwd().resolve()):
        raise ValueError("Путь выходит за пределы рабочей директории")

    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    photo_files = []
    try:
        for ext in extensions:
            photo_files.extend(Path(directory).glob(f'*{ext}'))
            photo_files.extend(Path(directory).glob(f'*{ext.upper()}')) # Учет регистра
        # Возвращаем список Path объектов
        return [f for f in photo_files if f.is_file()]
    except Exception as e:
        log_message(f"Ошибка при получении списка файлов: {e}")
        return []


def process_image(image_path):
    """
    Обработка фото: поворот, обрезка, резкость, яркость.
    Возвращает обработанное изображение PIL.Image.
    """
    try:
        with Image.open(image_path) as img:
            # Примеры базовой обработки
            # Поворот (например, поворот на 90 градусов)
            processed_img = img.rotate(360, expand=True)

#            numpy_img = np.array(img.convert('RGB'))

#            # Поворот кадра
###                current_frame_rotated = cv2.rotate(current_frame_captured, cv2.ROTATE_180)
#            processed_img = cv2.flip(numpy_img, -1)
#
#            # Обрезаем до 1280x720 если они больше
###                current_frame = self.center_crop(current_frame_rotated, 1280, 720)
#            current_frame = self.center_crop(current_frame_rotated, 1366, 768)
##                current_frame = self.center_crop(current_frame_captured, 1366, 768)
###                current_frame = self.center_crop(current_frame_rotated, 1600, 900)

            # Обрезка (например, обрезка до квадрата)
            # min_dimension = min(processed_img.size)
            # processed_img = processed_img.crop((0, 0, min_dimension, min_dimension))

            # Резкость (увеличение на 1.2)
            # enhancer = ImageEnhance.Sharpness(processed_img)
            # processed_img = enhancer.enhance(1.2)

            # Яркость (увеличение на 1.1)
            # enhancer = ImageEnhance.Brightness(processed_img)
            # processed_img = enhancer.enhance(1.1)

            # Возвращаем копию, если были изменения
            # return processed_img.copy()

            # 4. Преобразование результата обратно в PIL Image
#            processed_img = Image.fromarray(processed_img, mode='RGB')

            return processed_img # Возвращаем оригинальное изображение как пример

    except Exception as e:
        log_message(f"Ошибка при обработке изображения {image_path}: {e}")
        return None


def get_content_hash(image_path):
    """Вычислить хэш содержания файла."""
    try:
        with open(image_path, 'rb') as f:
            file_content = f.read()
            return hashlib.sha256(file_content).hexdigest()
    except Exception as e:
        log_message(f"Ошибка при вычислении хэша для {image_path}: {e}")
        return None

def log_message(message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {message}")

# Создайте функцию, которая кодирует файл и возвращает результат.
def encode_file(file_path):
    with open(file_path, "rb") as fid:
        file_content = fid.read()
    return base64.b64encode(file_content).decode("utf-8")

def perform_ocr(image_path, auth_manager, folder_id):
    """
    Распознавание текста.
    Заглушка: получить токен, вычислить хэш, запрос к OCR API.
    """
    data = {"mimeType": "image/png",
            "languageCodes": ["ru","en"],
            "content": encode_file(image_path)}
    #        "content": encode_file("/var/tmp/screens/original_20251021_113151_1.png")}

    url = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"

    # Получаем действительный токен через менеджер
    token = auth_manager.get_valid_token()
    
    headers= {"Content-Type": "application/json",
              "Authorization": f"Bearer {token}",
              "x-folder-id": folder_id,
              "x-data-logging-enabled": "true"}

    for attempt in range(3):
        try:
            w = requests.post(url=url, headers=headers, data=json.dumps(data), timeout=30)
            if w.status_code == 200:
                break
            time.sleep(1)
        except Exception as ex:
            log_message(f"OCR exception: {str(ex)}")
            if attempt == 2:
                raise
            time.sleep(3)

#    try:
#        w = requests.post(url=url, headers=headers, data=json.dumps(data), timeout=30)
#    except Exception as ex:
#        log_message(f"OCR exception: {str(ex)}")
#        return f"OCR error: {str(ex)}"
#
#    log_message(f"Status Code: {w.status_code}")
#
#    if w.status_code != 200:
#        return f"OCR error: {w.status_code} - {w.text}"
#
    try:
        response_json = w.json()
#        print("Response JSON:")
#        print(json.dumps(response_json, indent=2, ensure_ascii=False))

## DUMMY
#            print("Введите текст кейса (Ctrl+D или Ctrl+Z для завершения):")
#            lines = []
#            while True:
#                try:
#                    line = input()
#                except EOFError:
#                    break
#                lines.append(line)
#            response_json["result"]["textAnnotation"]["fullText"] = '\n'.join(lines)

#        # clear headers
##            pattern = r'^СБЕР\n|^УНИВЕРСИТЕТ\n|^НАЗАД\n|^ПРОПУСТИТЬ.*\n|^ЗАВЕРШИТЬ\n|Пройден.*\n|^Сбер Мини-МВА.*\n|^Прокторинг.*\n'
#        pattern = r'^СБЕР\n|.*ЕРСИТЕТ\n|^НАЗАД\n|^ПРОПУСТИТЬ.*\n|^ЗАВЕРШИТЬ\n|Пройден.*\n|.*Мини-МВА.*\n|^Прокторинг.*\n'
#        text_clear = re.sub(pattern, '', response_json["result"]["textAnnotation"]["fullText"],
#                            flags=re.IGNORECASE | re.MULTILINE)
#
#        return text_clear[:1500]
        return response_json["result"]["textAnnotation"]["fullText"][:1500]
    except json.JSONDecodeError as e:
        return f"OCR error: Ошибка JSON: {str(e)}"
    except Exception as e:
        return f"OCR error: {str(e)}"

    # Заглушка
    return ""
#    return f"Распознанный текст из {image_path.name}"


def save_results(processed_img, ocr_text, input_path, output_dir):
    """
    Сохранение обработанного изображения и текста.
    """
    try:
        # Создать имя файла для сохранения
        base_name = input_path.stem
        extension = input_path.suffix
        output_img_path = output_dir / f"{base_name}_processed{extension}"
        output_txt_path = output_dir / f"{base_name}_ocr.txt"

        # Сохранить изображение
        if processed_img:
            processed_img.save(output_img_path)
            log_message(f"Сохранено изображение: {output_img_path}")

        # Сохранить текст
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(ocr_text)
        log_message(f"Сохранен текст: {output_txt_path}")

    except Exception as e:
        log_message(f"Ошибка при сохранении результатов для {input_path}: {e}")


def main():
    """Основной цикл обработки."""
    args = parse_arguments()
    input_dir = Path(args.input_directory)

    if not input_dir.is_dir():
        log_message(f"Ошибка: Указанный путь не является каталогом: {input_dir}")
        sys.exit(1)

    # Определить выходной каталог
    output_dir = args.output_directory
    if output_dir is None:
        output_dir = input_dir / "processed_output"
    else:
        output_dir = Path(output_dir)

    # Берем пути из переменных окружения
    key_path = os.getenv('YANDEX_SERVICE_ACCOUNT_KEY_PATH', 'keys/authorized_key.json')
    auth_manager = YandexCloudAuthManager(key_path)
    folder_id = os.getenv('YANDEX_FOLDER_ID')

    if not all([folder_id]):
        raise ValueError("Не заданы обязательные переменные окружения: YANDEX_FOLDER_ID")

    # Создать выходной каталог, если его нет
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log_message(f"Ошибка при создании выходного каталога {output_dir}: {e}")
        sys.exit(1)

    log_message(f"Поиск файлов в: {input_dir}")
    log_message(f"Результаты будут сохранены в: {output_dir}")

    photo_files = get_photo_files(input_dir, {'.png'})

    if not photo_files:
        log_message(f"Фотографии не найдены в {input_dir}")
        return

    log_message(f"Найдено {len(photo_files)} файлов для обработки.")

    for photo_path in photo_files:
        log_message(f"Обработка файла: {photo_path.name}")
        try:
            # 1. Обработка фото
            processed_image = process_image(photo_path)
            if processed_image is None:
                log_message(f"[INFO] Пропуск обработки изображения для {photo_path.name}")
                continue # Переходим к следующему файлу

            # 2. Распознавание текста
            ocr_result = perform_ocr(photo_path, auth_manager, folder_id)

            # 3. Сохранение
            save_results(processed_image, ocr_result, photo_path, output_dir)

        except Exception as e:
            log_message(f"[CRIT] Критическая ошибка при обработке {photo_path.name}: {e}")
            # Продолжить обработку следующего файла
            continue
#        time.sleep(1.5)
#        break

    log_message("Обработка завершена.")


if __name__ == "__main__":
    main()
