import cv2
import numpy as np
import time
from PIL import Image, ImageFilter, ImageEnhance
import os
import base64
import requests
import json
#from gigachat import GigaChat
from langchain_gigachat.chat_models import GigaChat
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.tracers import ConsoleCallbackHandler
from telethon import TelegramClient, events
import asyncio
import re
import yandexcloud
from yandex.cloud.iam.v1.iam_token_service_pb2 import (CreateIamTokenRequest)
from yandex.cloud.iam.v1.iam_token_service_pb2_grpc import IamTokenServiceStub
import jwt
from datetime import datetime, timedelta
import threading
from dotenv import load_dotenv

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
        
class ScreenTextMonitor:
    def __init__(self, camera_index=0, similarity_threshold=0.90):
        """
        Инициализация монитора

        Args:
            camera_index: индекс камеры (0 - обычно встроенная камера)
            similarity_threshold: порог схожести изображений (0.95 = 95% схожести)
        """
        self.camera_index = camera_index
        self.similarity_threshold = similarity_threshold
        self.previous_frame = None
        self.frame_count = 0

        # Создание папки для сохранения изображений
        self.save_dir = "/var/tmp/screens"
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Изображения будут сохраняться в: {self.save_dir}")

        # Инициализация камеры
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise Exception(f"Не удалось подключиться к камере с индексом {self.camera_index}")

        # Настройка камеры для лучшего качества
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        # Устанавливаем размер буфера в 1 (самый новый кадр)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.search_query = ""
        # Берем пути из переменных окружения
        key_path = os.getenv('YANDEX_SERVICE_ACCOUNT_KEY_PATH', 'keys/authorized_key.json')
        self.auth_manager = YandexCloudAuthManager(key_path)
        
        self.session_name = "beep"

        self.api_id = os.getenv("TG_API_ID")
        self.api_hash = os.getenv("TG_API_HASH")
        self.folder_id = os.getenv('YANDEX_FOLDER_ID')

        if not all([self.api_id, self.api_hash, self.folder_id]):
            raise ValueError("Не заданы обязательные переменные окружения: TG_API_ID, TG_API_HASH, YANDEX_FOLDER_ID")

    def log_message(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {message}")

    def save_image(self, image, prefix="screen"):
        """Сохранение изображения в папку /var/tmp/screens"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}_{self.frame_count}.png"
        filepath = os.path.join(self.save_dir, filename)

        # Сохранение изображения
        cv2.imwrite(filepath, image)
        self.log_message(f"Изображение сохранено: {filepath}")
        return filepath

    # Создайте функцию, которая кодирует файл и возвращает результат.
    def encode_file(self, file_path):
      with open(file_path, "rb") as fid:
        file_content = fid.read()
      return base64.b64encode(file_content).decode("utf-8")

    def capture_frame(self, buffer_clear_frames=2):
        """Захват кадра с очисткой буфера"""
        # Очистка буфера если CAP_PROP_BUFFERSIZE не поддерживается
        for _ in range(buffer_clear_frames):
            self.cap.grab()

        ret, frame = self.cap.retrieve()
        if not ret:
            # Если retrieve не сработал, пробуем read
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("Не удалось захватить кадр с камеры")

        return frame

    # Функция для центрированной обрезки до целевого разрешения
    def center_crop(self, frame, target_width, target_height):
        height, width = frame.shape[:2]
        if width > target_width and height > target_height:
            start_x = (width - target_width) // 2
            start_y = (height - target_height) // 2
            end_x = start_x + target_width
            end_y = start_y + target_height
            return frame[start_y:end_y, start_x:end_x]
        return frame

    def compare_frames(self, frame1, frame2):
        """
        Сравнение двух кадров
        Возвращает коэффициент схожести (0-1)
        """
        if frame1 is None or frame2 is None:
            return 0.0

        # Обрезаем оба кадра до 1280x720 если они больше
        frame1_cropped = self.center_crop(frame1, 1280, 720)
        frame2_cropped = self.center_crop(frame2, 1280, 720)

        # Приведение к одинаковому размеру
        frame1_resized = cv2.resize(frame1_cropped, (640, 480))
        frame2_resized = cv2.resize(frame2_cropped, (640, 480))

        # Конвертация в grayscale для сравнения
        gray1 = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY)

        # Вычисление разницы
        diff = cv2.absdiff(gray1, gray2)
        similarity = 1.0 - (np.sum(diff) / (diff.size * 255.0))

        return similarity

    def preprocess_image(self, image):
        """
        Предобработка изображения для улучшения читаемости текста
        """
        # Конвертация OpenCV BGR в PIL RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # 1. Увеличение резкости
        sharpened = pil_image.filter(ImageFilter.UnsharpMask(
            radius=2,
            percent=150,
            threshold=3
        ))

        # 2. Увеличение контрастности
        contrast_enhancer = ImageEnhance.Contrast(sharpened)
        enhanced = contrast_enhancer.enhance(1.3)

        # 3. Увеличение резкости
        sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
        final_image = sharpness_enhancer.enhance(1.5)

        return final_image

    def extract_text_with_yandex(self, image):
        data = {"mimeType": "image/png",
                "languageCodes": ["ru","en"],
                "content": self.encode_file(image)}
        #        "content": encode_file("/var/tmp/screens/original_20251021_113151_1.png")}

        url = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"
 
        # Получаем действительный токен через менеджер
        token = self.auth_manager.get_valid_token()
        
        headers= {"Content-Type": "application/json",
                  "Authorization": f"Bearer {token}",
                  "x-folder-id": self.folder_id,
                  "x-data-logging-enabled": "true"}

        try:
            w = requests.post(url=url, headers=headers, data=json.dumps(data), timeout=30)
        except Exception as ex:
            self.log_message(f"OCR exception: {str(ex)}")
            return f"OCR error: {str(ex)}"

        self.log_message(f"Status Code: {w.status_code}")

        if w.status_code != 200:
            return f"OCR error: {w.status_code} - {w.text}"

        try:
            response_json = w.json()
#            print("Response JSON:")
#            print(json.dumps(response_json, indent=2, ensure_ascii=False))

# DUMMY
            print("Введите текст кейса (Ctrl+D или Ctrl+Z для завершения):")
            lines = []
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                lines.append(line)
            response_json["result"]["textAnnotation"]["fullText"] = '\n'.join(lines)

            # clear headers
#            pattern = r'^СБЕР\n|^УНИВЕРСИТЕТ\n|^НАЗАД\n|^ПРОПУСТИТЬ.*\n|^ЗАВЕРШИТЬ\n|Пройден.*\n|^Сбер Мини-МВА.*\n|^Прокторинг.*\n'
            pattern = r'^СБЕР\n|.*ЕРСИТЕТ\n|^НАЗАД\n|^ПРОПУСТИТЬ.*\n|^ЗАВЕРШИТЬ\n|Пройден.*\n|.*Мини-МВА.*\n|^Прокторинг.*\n'
            text_clear = re.sub(pattern, '', response_json["result"]["textAnnotation"]["fullText"], 
                                flags=re.IGNORECASE | re.MULTILINE)

            return text_clear[:1500]
        except json.JSONDecodeError as e:
            return f"OCR error: Ошибка JSON: {str(e)}"
        except Exception as e:
            return f"OCR error: {str(e)}"

    def string_to_int_array(self, text):
        try:
            return [int(x) for x in text.split()]
        except ValueError as e:
            self.log_message(f"Ошибка преобразования: {e}")
            return []

    def query_gigachat_reason(self, text):

        giga = GigaChat(
            credentials=os.environ["GIGACHAT_CREDENTIALS"],
            model="GigaChat-Max",
#            model="GigaChat",
            verify_ssl_certs=False,
            timeout=30,
        )   

        template = """Ты - студент, сдающий экзамен.
Перед тобой текст, распознанный со скриншота экрана монитора.
Необходимо ответить на экзаменационное задание, описанное в тексте (входные данные).
Для ответа на вопрос необходимо провести анализ, выполнить вычисления или построить логическую цепочку.
Экзаменационное задание сформулировано на русском языке. В тексте может быть небольшое количество английских символов, формул и терминов.
При формулировании ответа оцени все предложенные варианты ответа.

Формат ответа: Суммаризация рассуждений объёмом не более 3000 символов.
{context}
Экзаменационное задание: {input}"""

#Контекст: {context}

        prompt = ChatPromptTemplate.from_template(template)
        question_answer_chain = create_stuff_documents_chain(giga, prompt)

        print(self.search_query)
        rag_chain = (
            {
                "input": lambda x: x["input"],
                "context": lambda x: []  # Пустой контекст
#                "context": RunnableLambda(lambda x: self.search_query) | retriever_pres
            }
            | question_answer_chain
        )

        result = ""
        try:
        # Вариант 3: Универсальный
#            result = rag_chain.invoke({"input": text}, config={"callbacks": [ConsoleCallbackHandler()]})
            result = rag_chain.invoke({"input": text})
        except Exception as e:
            self.log_message(f"Ошибка запроса к LLM: {e}")
        reason = result if not isinstance(result, dict) else result.get("output", result)
        print(f"Суммаризация рассуждений: {reason}")


        # Промпт для классификации задания - должен быть ChatPromptTemplate
        reason_template = """Ты - студент, сдающий экзамен.
Перед тобой текст, распознанный со скриншота экрана монитора.
Возможно, в тексте кроме самого экзаменационного задания есть паразитные слова и символы, которые надо проигнорировать.
Возможно, в тексте есть название кнопок навигации по тесту, такие как Продолжить, Назад, Ответить, Завершить.

Экзаменационное задание сформулировано на русском языке. В тексте может быть небольшое количество английских символов, формул и терминов.
В самом задании (во входных данных) могут быть указаны следующие данные: 
Описание задачи; Несколько вариантов ответов, каждый с новой строки или пронумерованные 1, 2, 3, 4, 5 и т.д. или A, B, C, D и т.д.;

В запросе также приведены твои собственные рассуждения по этой задаче.
Необходимо ответить на экзаменационное задание. 
При выборе вариантов ответа опирайся в первую очередь на свои собственные рассуждения.

Формат ответа: В ответе должна быть представлена валидная строка JSON с двойными кавычками.
Каждый элемент JSON строки содержит в качестве ключа цифру - номер правильного ответа по очереди и текстовое описание этого варианта.
В случае нескольких правильных ответов, необходимо сформировать строку с номерами ответов и их текстовыми описаниями.

Например: Если это первый (верхний) по очереди ответ или ответ А, выведи {{"1":"описание первого ответа"}}.
Если второй или В выведи {{"2":"описание второго ответа"}}, и т.д.
В JSON строке ключ - это всегда цифра, а описание в точности воспроизводит один из вариантов ответа без дополнений

Если по условиям задачи необходимо указать более одного ответа (множественный выбор), выведи все правильные варианты.
Например: {{"1":"описание первого ответа","2":"описание второго ответа"}}

Ограничения:
- В ответах не может быть более восьми вариантов. Если ты распознал больше восьми вариантов ответа, выведи {{"0":"Ошибка"}}
- Если в задании не указано, что требуется выбрать более одного варианта ответа, всегда выбирай только один правильный.
- Если в задании точно указано количество верных вариантов, выбирай именно такое количество правильных ответов.
- Ключ варианта ответа всегда цифра, не более 8
- Если текст задания пустой, выведи {{"0":"Ошибка"}}
- Если не удаётся найти ответ, выведи {{"0":"Ошибка"}}
- Если задание не сформулировано или распознано не полностью, выведи {{"0":"Ошибка"}}
- Если получился вариант ответа больше 8 (восьми) выведи {{"0":"Ошибка"}}
- Никогда не выводи вариант ответа больше 8 восьми

Не давай никаких дополнительных комментариев. 

Экзаменационное задание: {input}

Твои размышления по этому вопросу: {reason}"""

        prompt = ChatPromptTemplate.from_template(reason_template)

        # Создаем цепочку для классификации
        reasoning_chain = prompt | giga

        result = ""
        try:
            # Шаг 2: Классифицируем используя ТОЛЬКО search_query
#            result = reasoning_chain.invoke({"reason": reason, "input": text}, config={"callbacks": [ConsoleCallbackHandler()]})
            result = reasoning_chain.invoke({"reason": reason, "input": text})
        except Exception as e:
            self.log_message(f"Ошибка запроса к LLM: {e}")

        # Простое извлечение - всегда работает
        rag = getattr(result, 'content', str(result))

        try: 
            # Очищаем строку от лишних пробелов и переносов
            cleaned_response = rag.strip().replace('\n', '').replace('  ', ' ')
            response_json = json.loads(cleaned_response)
            return response_json
        except json.JSONDecodeError as e:
            # Альтернативная попытка - найти JSON в тексте
            import re
            json_match = re.search(r'\{.*\}', rag)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            return {"raw_response": rag}
        except Exception as e:
            self.log_message(f"Другая ошибка: {e}")
            return {"error": f"Ошибка при обработке ответа"}
        

    def query_gigachat_rag(self, text):

        giga = GigaChat(
            credentials=os.environ["GIGACHAT_CREDENTIALS"],
            model="GigaChat-Max",
#            model="GigaChat",
            verify_ssl_certs=False,
            timeout=30,
        )   

        template = """Ты - студент, сдающий экзамен.
Перед тобой текст, распознанный со скриншота экрана монитора.
Необходимо ответить на экзаменационное задание, описанное в тексте (входные данные).
Возможно, в тексте кроме самого экзаменационного задания есть паразитные слова и символы, которые надо проигнорировать.
Возможно, в тексте есть название кнопок навигации по тесту, такие как Продолжить, Назад, Ответить, Завершить.

Экзаменационное задание сформулировано на русском языке. В тексте может быть небольшое количество английских символов, формул и терминов.
В самом задании (во входных данных) могут быть указаны следующие данные: 
Описание задачи; Несколько вариантов ответов, каждый с новой строки или пронумерованные 1, 2, 3, 4, 5 и т.д. или A, B, C, D и т.д.;

Формат ответа: В ответе должна быть представлена валидная строка JSON с двойными кавычками.
Каждый элемент JSON строки содержит в качестве ключа цифру - номер правильного ответа по очереди и текстовое описание этого варианта.
В случае нескольких правильных ответов, необходимо сформировать строку с номерами ответов и их текстовыми описаниями.

Например: Если это первый (верхний) по очереди ответ или ответ А, выведи {{"1":"описание первого ответа"}}.
Если второй или В выведи {{"2":"описание второго ответа"}}, и т.д.
В JSON строке ключ - это всегда цифра, а описание в точности воспроизводит один из вариантов ответа без дополнений

Если по условиям задачи необходимо указать более одного ответа (множественный выбор), выведи все правильные варианты.
Например: {{"1":"описание первого ответа","2":"описание второго ответа"}}

Ограничения:
- В ответах не может быть более восьми вариантов. Если ты распознал больше восьми вариантов ответа, выведи {{"0":"Ошибка"}}
- Если в задании не указано, что требуется выбрать более одного варианта ответа, всегда выбирай только один правильный.
- Если в задании точно указано количество верных вариантов, выбирай именно такое количество правильных ответов.
- Ключ варианта ответа всегда цифра, не более 8
- Если текст задания пустой, выведи {{"0":"Ошибка"}}
- Если не удаётся найти ответ, выведи {{"0":"Ошибка"}}
- Если задание не сформулировано или распознано не полностью, выведи {{"0":"Ошибка"}}
- Если получился вариант ответа больше 8 (восьми) выведи {{"0":"Ошибка"}}
- Никогда не выводи вариант ответа больше 8 восьми

Не давай никаких дополнительных комментариев. 
{context}
Экзаменационное задание: {input}"""

#Контекст: {context}

        prompt = ChatPromptTemplate.from_template(template)
        question_answer_chain = create_stuff_documents_chain(giga, prompt)

#        print(self.search_query)
        rag_chain = (
            {
                "input": lambda x: x["input"],
                "context": lambda x: []  # Пустой контекст
#                "context": RunnableLambda(lambda x: self.search_query) | retriever_pres
            }
            | question_answer_chain
        )

        result = ""
        try:
            # Вариант 3: Универсальный
#            result = rag_chain.invoke({"input": text}, config={"callbacks": [ConsoleCallbackHandler()]})
            result = rag_chain.invoke({"input": text})
        except Exception as e:
            self.log_message(f"Ошибка запроса к LLM: {e}")

        rag = result if not isinstance(result, dict) else result.get("output", result)
#
#        try:
#            response_content = rag
#            response_json = json.loads(response_content)
#            return response_json
#        except json.JSONDecodeError as e:
#            return f"Ошибка при распознавании текста от Гигачат"
#        except Exception as e:
#            self.log_message(f"Другая ошибка: {e}")
#            return f"Ошибка при обработке ответа"

        try: 
            # Очищаем строку от лишних пробелов и переносов
            cleaned_response = rag.strip().replace('\n', '').replace('  ', ' ')
            response_json = json.loads(cleaned_response)
            return response_json
        except json.JSONDecodeError as e:
            # Альтернативная попытка - найти JSON в тексте
            import re
            json_match = re.search(r'\{.*\}', rag)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            return {"raw_response": rag}
        except Exception as e:
            self.log_message(f"Другая ошибка: {e}")
            return {"error": f"Ошибка при обработке ответа"}

    def send_capture_sync(self, image_path, **kwargs):
        """
        Синхронный метод для отправки изображений с обработкой асинхронного контекста
        """
        # Проверяем существование файла перед отправкой
        if not os.path.exists(image_path):
            self.log_message(f"Ошибка: файл не существует - {image_path}")
            return

        async def async_wrapper():
            async with TelegramClient(self.session_name, self.api_id, self.api_hash) as client:
                self.client = client
                await self._send_capture_async(image_path, **kwargs)
        
        try:
            # Пытаемся запустить новый event loop
            asyncio.run(async_wrapper())
        except RuntimeError as e:
            # Если уже есть запущенный event loop (например, в Jupyter или другом async контексте)
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                self.log_message("Обнаружен работающий event loop, используем его")
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Запускаем в существующем loop
                    loop.create_task(async_wrapper())
                else:
                    loop.run_until_complete(async_wrapper())
            else:
                raise

    async def _send_capture_async(self, image, recipient='LinuxGodsWorkaholicBot'):

        """
        Асинхронная реализация отправки уведомлений
        """
        try:
            entity = await self.client.get_entity(recipient)

            await self.client.send_file(entity, image)
            self.log_message(f"Отправлено capture")
                            
        except Exception as e:
            self.log_message(f"Ошибка: {e}")
   
    def send_notifications_sync(self, answers, **kwargs):
        """
        Синхронный метод для отправки уведомлений
        """
        async def async_wrapper():
            async with TelegramClient(self.session_name, self.api_id, self.api_hash) as client:
                self.client = client
                await self._send_notifications_async(answers, **kwargs)
        
        asyncio.run(async_wrapper())
   
    async def _send_notifications_async(self, answers, recipient='LinuxGodsWorkaholicBot',
                                      delay_between_messages=3, delay_between_numbers=10):

        """
        Асинхронная реализация отправки уведомлений
        """
        try:
            entity = await self.client.get_entity(recipient)

            # Сначала отправляем общее количество нотификаций
            total_notifications = len(answers.keys())
        
            # Отправляем количество только если нотификаций больше одной
            if total_notifications > 1:

                count_message = f"Количество уведомлений: {total_notifications}"
                for message_num in range(total_notifications):
                    await self.client.send_message(entity, count_message)
                    self.log_message(f"Отправлено сообщение: {count_message}")
                            
                    if message_num < total_notifications - 1:
                        await asyncio.sleep(delay_between_messages)

                # Пауза 10 секунд
                self.log_message(f"Ожидание {delay_between_numbers} сек перед отправкой уведомлений...")
                await asyncio.sleep(delay_between_numbers)

            for i, (number, description) in enumerate(answers.items()):
                self.log_message(f"Отправка ответа {number}: {description}")
                            
                for message_num in range(int(number)):  # преобразуем строку в число для счетчика
                    message = f"{number}: {description}"
                    await self.client.send_message(entity, message)
                    self.log_message(f"Отправлено сообщение {message_num + 1}/{number}: {description}")
                            
                    if message_num < int(number) - 1:
                        await asyncio.sleep(delay_between_messages)

                if i < len(answers) - 1:
                    self.log_message(f"Ожидание {delay_between_numbers} сек...")
                    await asyncio.sleep(delay_between_numbers)

            self.log_message("Все уведомления отправлены!")

        except Exception as e:
            self.log_message(f"Ошибка: {e}")


    def query_gigachat_task_type(self, text):
        # Инициализируем массив для хранения хэшей, если ещё не инициализирован
        if not hasattr(self, '_seen_task_hashes'):
            self._seen_task_hashes = set()
        
        giga = GigaChat(
            credentials=os.environ["GIGACHAT_CREDENTIALS"],
            model="GigaChat-Max",
#            model="GigaChat",
            verify_ssl_certs=False,
            timeout=30,
        )

        # Промпт для извлечения поискового запроса
        query_extraction_prompt = ChatPromptTemplate.from_template("""
Перед тобой текст, распознанный со скриншота (снимка экрана).
Текст содержит экзаменационное задание с несколькими вариантами ответа.
Кроме этого в тексте могут встречаться всевозможные паразитные символы, которые были распознаны ошибочно.
Также в тексте могут быть название темы и название дисциплины или предметной области.
Возможно, в тексте есть название кнопок навигации по тесту, такие как Продолжить, Назад, Ответить, Завершить.
Твоя задача выделить из всего текста только полное описание экзаменационного задания без ответов.
Не добавляй никаких дополнительных пояснений и корректировок.

Формат ответа: Распознанное экзаменационное задание без вариантов ответа

Далее идёт распознанный текст, который необходимо обработать: {input} """)

        # Цепочка для извлечения запроса
        query_extraction_chain = query_extraction_prompt | giga

        def extract_with_llm(full_input):
            """Использует LLM для извлечения поискового запроса"""
            text = full_input["input"]
            search_query = ""
            try:
                search_query = query_extraction_chain.invoke({"input": text}).content
            except Exception as e:
                self.log_message(f"Ошибка запроса к LLM: {e}")
            print(f"🔍 Извлеченный поисковый запрос: '{search_query}'")
            return search_query

        # Шаг 1: Извлекаем search_query
        self.search_query = extract_with_llm({"input": text})
        
        # Шаг 2: Создаём хэш извлечённого задания
        import hashlib
        task_hash = hashlib.md5(self.search_query.strip().encode()).hexdigest()
        print(f"📝 Хэш задания: {task_hash[:8]}...")
        
        # Шаг 3: Проверяем, видели ли уже это задание
        if task_hash in self._seen_task_hashes:
            print(f"🔄 Задание уже встречалось ранее. Возвращаем тип 2 (анализ)")
            # Добавляем в логи для отладки
            self.log_message(f"Повторное задание, хэш: {task_hash[:8]}..., принудительно тип 2")
            return "2"
        
        # Шаг 4: Если задание новое - сохраняем хэш и продолжаем классификацию
        self._seen_task_hashes.add(task_hash)
        print(f"💾 Новое задание, сохранён хэш: {task_hash[:8]}...")
        
        # Промпт для классификации задания
        classification_prompt = ChatPromptTemplate.from_template(""" 
Классифицируй экзаменационное задание.
Выбери один из вариантов: 
1 - Проверка знания определённого факта
2 - Анализ, выведение или вычисление ответа в результате рассуждения.
Если для нахождения ответа не требуется строить логическую цепочку или выполнять вычисления, значит задание первого типа. 
Если же необходимо что-то проанализировать или вычислить - второго.

Формат ответа: Тип задания (только цифра 1 или 2).

Текст задания: {search_query}""")

        # Создаем цепочку для классификации
        classification_chain = classification_prompt | giga

        result = "2" 
        try:
            # Шаг 5: Классифицируем задание через LLM
            result = classification_chain.invoke({"search_query": self.search_query})
        except Exception as e:
            self.log_message(f"Ошибка запроса к LLM: {e}")

        final_result = result.content if hasattr(result, 'content') else result
        
        print(f"📊 Тип задания: {final_result}")
        return final_result

    def optimize_image_for_send(self, image_path, scale_factor=0.25, quality=60):
        """
        Оптимизация изображения: уменьшение размера и сжатие
        
        Args:
            image_path: путь к оригинальному изображению
            scale_factor: коэффициент масштабирования (0.25 = в 4 раза меньше)
            quality: качество JPEG (1-100)
        
        Returns:
            Путь к оптимизированному изображению
        """
        try:
            # Загрузка изображения
            img = cv2.imread(image_path)
            if img is None:
                self.log_message(f"Ошибка загрузки изображения: {image_path}")
                return image_path  # Возвращаем оригинал в случае ошибки
            
            # Получаем оригинальные размеры
            height, width = img.shape[:2]
            self.log_message(f"Оригинальный размер: {width}x{height}")
            
            # Уменьшаем в 4 раза (scale_factor=0.25)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Масштабируем с интерполяцией для сохранения читаемости
            optimized_img = cv2.resize(img, (new_width, new_height), 
                                       interpolation=cv2.INTER_AREA)
            
            # Создаем путь для оптимизированного изображения
            orig_dir = os.path.dirname(image_path)
            orig_filename = os.path.basename(image_path)
            name_without_ext, ext = os.path.splitext(orig_filename)
            optimized_filename = f"optimized_{name_without_ext}.jpg"
            optimized_path = os.path.join(orig_dir, optimized_filename)
            
            # Сохраняем с максимальным сжатием (низкое качество JPEG)
            cv2.imwrite(optimized_path, optimized_img, 
                        [cv2.IMWRITE_JPEG_QUALITY, quality])
            del optimized_img  # Освобождаем память
            
            # Сравниваем размеры файлов
            orig_size = os.path.getsize(image_path) / 1024  # в КБ
            opt_size = os.path.getsize(optimized_path) / 1024
            compression_ratio = orig_size / opt_size if opt_size > 0 else 0
            
            self.log_message(f"Оптимизированный размер: {new_width}x{new_height}")
            self.log_message(f"Размер файла: {orig_size:.1f}КБ → {opt_size:.1f}КБ (сжатие в {compression_ratio:.1f} раз)")
            
            return optimized_path
            
        except Exception as e:
            self.log_message(f"Ошибка оптимизации изображения: {str(e)}")
            return image_path  # Возвращаем оригинал в случае ошибки

    def run_monitoring(self):
        """Основной цикл мониторинга"""
        print("Запуск мониторинга...")
        print("Для остановки нажмите Ctrl+C")

        try:
            while True:
                start_time = time.time()

                # Захват текущего кадра
                current_frame_captured = self.capture_frame()

                # Поворот кадра
##                current_frame_rotated = cv2.rotate(current_frame_captured, cv2.ROTATE_180)
                current_frame_rotated = cv2.flip(current_frame_captured, -1)

                # Обрезаем до 1280x720 если они больше
##                current_frame = self.center_crop(current_frame_rotated, 1280, 720)
                current_frame = self.center_crop(current_frame_rotated, 1366, 768)
#                current_frame = self.center_crop(current_frame_captured, 1366, 768)
##                current_frame = self.center_crop(current_frame_rotated, 1600, 900)

                self.frame_count += 1

                print(f"\n--- Кадр #{self.frame_count} ---")

                # Сравнение с предыдущим кадром
                if self.previous_frame is not None:
                    similarity = self.compare_frames(self.previous_frame, current_frame)
                    print(f"Схожесть с предыдущим кадром: {similarity:.2%}")

                    # Если изменения незначительные, пропускаем обработку
                    if similarity > self.similarity_threshold:
                        self.save_image(current_frame, "similar")
                        print("Изменения незначительные, пропускаем обработку")
                        self.previous_frame = current_frame
# DUMMY                        
#                        time.sleep(20 - (time.time() - start_time))
#                        continue

                # Обработка изображения
                print("Обнаружены значительные изменения, обрабатываем изображение...")

                # Сохранение оригинального изображения
                orig_image = self.save_image(current_frame, "original")
#                orig_image = "images/Screenshot 2025-10-23 09-58-38.png"

                # Основной вариант (рекомендуемый)
                optimized_image_path = self.optimize_image_for_send(orig_image,
                                                    scale_factor=0.5,
                                                    quality=75)  # Еще сильнее сжимаем

                self.send_capture_sync(optimized_image_path)

#                processed_image = self.preprocess_image(current_frame)

                # Распознавание текста
                self.log_message("Распознавание текста...")
#                text = self.extract_text_with_moondream2(processed_image)
                text = self.extract_text_with_yandex(orig_image)

                if text is None:
                    # Обработка случая, когда OCR вернул None (произошла ошибка)
                    print("OCR завершился с ошибкой")
                    continue
                elif len(text) == 0:
                    # Обработка случая, когда ничего не распознано вообще, закрыта крышка?
                    print("Нет текста для распознавания")
                    # Сохраняем фрейм, так как в нём нет ничего плохoго
                    self.previous_frame = current_frame
                    continue
                elif isinstance(text, str) and "OCR error" in text:
                    # Обработка случая, когда OCR вернул строку с ошибкой
                    print("Обнаружена ошибка OCR в тексте")
                    continue

                # Вывод результата
                print("\n" + "="*50)
                print("РАСПОЗНАННЫЙ ТЕКСТ:")
                print("="*50)
                print(text)
                print("="*50)

                self.log_message("Запрос модели...")
                answer = self.query_gigachat_task_type(text)

                # factual
                if answer == "1":
#                    answer = self.query_gigachat_reason(text)
                    answer = self.query_gigachat_rag(text)

                else:
                    answer = self.query_gigachat_reason(text)

                # Вывод результата
                print("\n" + "="*50)
                print("ОТВЕТ МОДЕЛИ RAG:")
                print("="*50)
                print(answer)
                print("="*50)

                # Отправляем уведомления
                self.log_message("\nОтправка уведомлений...")
## DUMMY
#                self.send_notifications_sync(
##                    numbers=numbers,
#                    answers=answer,
#                    recipient='LinuxGodsWorkaholicBot',
#                    delay_between_messages=3,
#                    delay_between_numbers=7
#                )

                # Сохранение текущего кадра как предыдущего
                self.previous_frame = current_frame

                # Ожидание до следующей итерации
                elapsed_time = time.time() - start_time
                if elapsed_time < 20:
                    sleep_time = 20 - elapsed_time
                    self.log_message(f"Ожидание {sleep_time:.1f} секунд до следующего кадра...")
                    time.sleep(sleep_time)
                else:
                    self.log_message("Обработка заняла больше 20 секунд, переходим к следующему кадру немедленно")

        except KeyboardInterrupt:
            print("\nМониторинг остановлен пользователем")
        except Exception as e:
            print(f"Произошла ошибка: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Очистка ресурсов"""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        self.log_message("Ресурсы освобождены")

def main():
    # Настройки
    CAMERA_INDEX = 0  # 0 - обычно встроенная камеры, 1 - внешняя камера
    SIMILARITY_THRESHOLD = 0.995  # 95% схожести

    try:
        monitor = ScreenTextMonitor(
            camera_index=CAMERA_INDEX,
            similarity_threshold=SIMILARITY_THRESHOLD
        )
        monitor.run_monitoring()
    except Exception as e:
        print(f"Ошибка инициализации: {str(e)}")
        print("Проверьте:")
        print("1. Подключена ли камера")
        print("2. Правильный ли индекс камеры")
        print("3. Установлены ли все зависимости")

if __name__ == "__main__":
    main()
