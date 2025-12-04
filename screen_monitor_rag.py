import cv2
import numpy as np
import time
from PIL import Image, ImageFilter, ImageEnhance
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from datetime import datetime
import base64
import requests
import json
from gigachat import GigaChat
from telethon import TelegramClient, events
import asyncio
import re
from chromadb.config import Settings
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings
from langchain_chroma import Chroma


class ScreenTextMonitor:
    def __init__(self, camera_index=0, similarity_threshold=0.90, api_id="25315069", api_hash='419b7cd9f055a855ffd2f06948ab882e', session_name='beep'):
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
#        # Загрузка модели Moondream2
#        print("Загрузка модели Moondream2...")
#        self.model_id = "vikhyatk/moondream2"
#        self.device = "cuda" if torch.cuda.is_available() else "cpu"
#        print(f"Используется устройство: {self.device}")
#
#        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
#        self.model = AutoModelForCausalLM.from_pretrained(
#            self.model_id,
#            trust_remote_code=True,
#            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
#        ).to(self.device)
#        print("Модель загружена успешно!")

        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
#        self.client = TelegramClient(session_name, api_id, api_hash)

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

#    def capture_frame(self):
#        """Захват кадра с камеры"""
#        ret, frame = self.cap.read()
#        if not ret:
#            raise Exception("Не удалось захватить кадр с камеры")
#        return frame

#    def compare_frames(self, frame1, frame2):
#        """
#        Сравнение двух кадров
#        Возвращает коэффициент схожести (0-1)
#        """
#        if frame1 is None or frame2 is None:
#            return 0.0
#
#        # Приведение к одинаковому размеру
#        frame1_resized = cv2.resize(frame1, (640, 480))
#        frame2_resized = cv2.resize(frame2, (640, 480))
#
#        # Конвертация в grayscale для сравнения
#        gray1 = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2GRAY)
#        gray2 = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY)
#
#        # Вычисление разницы
#        diff = cv2.absdiff(gray1, gray2)
#        similarity = 1.0 - (np.sum(diff) / (diff.size * 255.0))
#
#        return similarity
#
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
        #        "content": encode_file("images/8930.jpg")}

        url = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"

        token = ""

        headers= {"Content-Type": "application/json",
                  "Authorization": "Bearer {:s}".format(token),
                  "x-folder-id": "b1ghg3qttqeg3e6qpgp5",
                  "x-data-logging-enabled": "true"}

        w = requests.post(url=url, headers=headers, data=json.dumps(data), timeout=30)

        self.log_message(f"Status Code: {w.status_code}")

        if w.status_code != 200:
            return f"OCR error: {w.status_code} - {w.text}"

        try:
            response_json = w.json()
#            print("Response JSON:")
#            print(json.dumps(response_json, indent=2, ensure_ascii=False))

# DUMMY
#            print("Введите текст кейса (Ctrl+D или Ctrl+Z для завершения):")
#            lines = []
#            while True:
#                try:
#                    line = input()
#                except EOFError:
#                    break
#                lines.append(line)
#            response_json["result"]["textAnnotation"]["fullText"] = '\n'.join(lines)


#            response_json["result"]["textAnnotation"]["fullText"] = """
#СБЕР
#УНИВЕРСИТЕТ
#Сбер Мини-МВА 16 поток Экзамен по дисциплине "Комму
#Что относится к источникам переговорной силы (мночественный выбор):
#Сила коалиции
#B
#Сила стратегии
#Сила ситуации
#D
#Сила личности
#Сила здравого смысла
#Сила давления
#"""

#СБЕР
#УНИВЕРСИТЕТ
#Сбер Мини-МВА 16 поток Экзамен по дисциплине "Комму
#На шаге – «Договаривайтесь» полезно (множественный выбор):
#Потребовать финальную уступку от визави
#Понижать НАОС визави
#Определять приоритеты
#Резюмировать итоги
#Зафиксировать договоренности

#СБЕР
#УНИВЕРСИТЕТ
#Сбер Мини-МВА 16 поток Экзамен по дисципине "Коммуњ
#Переговорная карта может быть (множественный выбор):
#Тактическая
#B
#Пестрая
#Наивная
#Комплексная
#"""


            # clear headers
            pattern = r'^СБЕР\n|^УНИВЕРСИТЕТ\n|^НАЗАД\n|^ПРОПУСТИТЬ.*\n|^ЗАВЕРШИТЬ\n|Пройден.*\n|^Сбер Мини-МВА.*\n|^Прокторинг.*\n'
            text_clear = re.sub(pattern, '', response_json["result"]["textAnnotation"]["fullText"], 
                                flags=re.IGNORECASE | re.MULTILINE)

            return text_clear[:1500]
        except json.JSONDecodeError as e:
            return f"Ошибка JSON: {str(e)}"
        except Exception as e:
            return f"Ошибка при распознавании: {str(e)}"

    def string_to_int_array(self, text):
        try:
            return [int(x) for x in text.split()]
        except ValueError as e:
            self.log_message(f"Ошибка преобразования: {e}")
            return []


    def query_gigachat_rag(self, text):

        from langchain_gigachat.chat_models import GigaChat

        giga = GigaChat(
            credentials=os.environ["GIGACHAT_CREDENTIALS"],
            model="GigaChat-Max",
#            model="GigaChat",
            verify_ssl_certs=False,
            timeout=1200,
        )   
        # Используем те же эмбеддинги, что и при создании базы
        embeddings = GigaChatEmbeddings(
            credentials=os.environ["GIGACHAT_CREDENTIALS"],
            verify_ssl_certs=False
        )

        # Используем те же эмбеддинги, что и при создании базы
        embeddings_pres = GigaChatEmbeddings(
            credentials=os.environ["GIGACHAT_CREDENTIALS"],
            verify_ssl_certs=False
        )

        persist_directory = "./chroma_db_l_web"
#        persist_directory = "./chroma_db_c_pres"

        # Загружаем базу
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

        try:
            # Получаем все чанки из базы
            all_data = db.get()

            if all_data and 'documents' in all_data and len(all_data['documents']) > 0:
                total_chunks = len(all_data['documents'])
                print(f"✅ Всего чанков в базе: {total_chunks}")
#                print("="*50)
            else:
                print("❌ В базе нет чанков или база пуста")

        except Exception as e:
            print(f"❌ Ошибка при получении чанков: {e}")

        persist_directory_pres = "./chroma_db_l_pres"

        # Загружаем базу
        db_pres = Chroma(
            persist_directory=persist_directory_pres,
            embedding_function=embeddings_pres
        )

        try:
            # Получаем все чанки из базы
            all_data = db_pres.get()

            if all_data and 'documents' in all_data and len(all_data['documents']) > 0:
                total_chunks = len(all_data['documents'])
                print(f"✅ Всего чанков в базе pres: {total_chunks}")
#                print("="*50)
            else:
                print("❌ В базе нет чанков или база пуста")

        except Exception as e:
            print(f"❌ Ошибка при получении чанков: {e}")

        # RAG interact
#        retriever = db.as_retriever()
#        retriever = db.as_retriever(
#            search_kwargs={"k": 3}  # использовать 3 чанка
#        )
        retriever = db.as_retriever(
            search_kwargs={"k": 3}  # использовать 3 чанка
        )
        retriever_pres = db_pres.as_retriever(
            search_kwargs={"k": 5}  # использовать 3 чанка
        )
        from langchain.retrievers import EnsembleRetriever
        # Создаем ансамблевый ретривер
        ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever, retriever_pres],
            weights=[0.3, 0.7]  # веса для каждого ретривера
        )

        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate
        from langchain.callbacks.tracers import ConsoleCallbackHandler

        template = """Ты - студент, сдающий экзамен по дисциплине Лидерство и управление людьми. 
Перед тобой текст, распознанный со скриншота экрана монитора.
Необходимо ответить на экзаменационное задание, описанное в тексте (входные данные).
Возможно, в тексте кроме самого экзаменационного задания есть паразитные слова и символы, которые надо проигнорировать.
Экзаменационное задание сформулировано на русском языке. В тексте может быть небольшое количество английских символов, формул и терминов.
В самом задании (во входных данных) могут быть указаны следующие данные: Описание задачи; Несколько вариантов ответов, 
каждый с новой строки или пронумерованные 1, 2, 3, 4, 5 и т.д. или A, B, C, D и т.д.
Формат ответа: В ответе должна быть представлена валидная строка JSON с двойными кавычками.
Каждый элемент JSON строки содержит номер правильного ответа и текстовое описание этого варианта.
В случае нескольких правильных ответов, необходимо сформировать строку с номерами ответов и их текстовыми описаниями.
Например: Если это первый (верхний) по очереди ответ или ответ А, выведи {{"1":"описание первого ответа"}}.
Если второй или В выведи {{"2":"описание второго ответа"}}, и т.д.
В JSON строке ключ - это всегда число, а описание в точности воспроизводит один из вариантов ответа без дополнений
Не давай никаких дополнительных комментариев. 
Если по условиям задачи необходимо указать более одного ответа (множественный выбор), выведи все правильные варианты.
Например: {{"1":"описание первого ответа","2":"описание второго ответа"}}
Ограничения:
- В ответах не может быть более восьми вариантов. Если ты распознал больше восьми вариантов ответа, выведи {{"0":"Ошибка"}}
- Если в задании не указано, что требуется выбрать более одного варианта ответа, всегда выбирай только один правильный.
- Если в задании точно указано количество верных вариантов, выбирай именно такое количество правильных ответов.
- Если текст задания пустой, выведи {{"0":"Ошибка"}}
- Если не удаётся найти ответ, выведи {{"0":"Ошибка"}}
- Если задание не сформулировано или распознано не полностью, выведи {{"0":"Ошибка"}}
- Если получился вариант ответа больше 8 (восьми) выведи {{"0":"Ошибка"}}
- Никогда не выводи вариант ответа больше 8 восьми

Контекст: {context}

Вопрос пользователя: {input}"""

        prompt = ChatPromptTemplate.from_template(template)

        question_answer_chain = create_stuff_documents_chain(giga, prompt)
#        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        rag_chain = create_retrieval_chain(ensemble_retriever, question_answer_chain)

        #q = test_query
        #pure = giga.invoke([HumanMessage(content=q)]).content
        #print(f"Pure: {pure}")

        rag = rag_chain.invoke({"input": text})["answer"]
#        rag = rag_chain.invoke({"input": text},config={"callbacks": [ConsoleCallbackHandler()]})["answer"]
#        print(f"RAG: {rag}")

        try:
            response_content = rag
            response_json = json.loads(response_content)
            return response_json
        except json.JSONDecodeError as e:
            self.log_message(f"Ошибка парсинга JSON: {e}")
            self.log_message(f"Полученный текст: {response_content}")
            return f"Ошибка при распознавании текста от Гигачат"
        except Exception as e:
            self.log_message(f"Другая ошибка: {e}")
            return f"Ошибка при обработке ответа"



    def query_gigachat(self, text):

        giga = GigaChat(
           credentials="ZGIyMTNkN2QtNjBlMy00OWM3LWI3OTQtNWM5MjliYTk1N2E2OmI4MDNlNThmLTdhNDMtNDE1Yy1hMTU2LWRjYTFiYzJmODdhMQ==",
#           model="GigaChat",
           model="GigaChat-Max",
        )

#        text = "\n\nСбер Мини-МВА 16 поток. Финансы в новой экономике. Экзамен\n\nТема: Краткосрочное решение на основе издержек и выручки.\n\nРесторан реализует 60 блюд по 2.5 руб при этом совокупные издержки составляют 130 руб. из них постоянные - 40 руб. Что делать менеджеру ресторана? Выберите два варианта ответа.\n\nA Продолжать, так как переменные издержки покрываются, а убыток ограничен постоянными \nB Закрыть так как цена ниже переменных затрат\nC Немедленно увеличить аренду\nD Продолжать работу, так как покрываются все издержки."

# GigaChat-Max can do it
#        text = "асть 2\
#Посчитайте стандартное отклонение для ряда\
#[1,2,3,4,3,3,6,7]. Выберите ближайшее число к\
#вашему ответу число.\
#*\
#1 балл\
#2.31\
#1.86\
#3.62\
#3.1"

#        text = "Какие из перечисленных метрик, являются метриками для задачи\
#регрессии? Нужно выбрать несколько вариантов ответов.\
#1 ба\
#MSE\
#MAE\
#Precision\
#Recall\
#accuracy\
#Козффициент детерминации R2"

#        text = "асть 6\
#Допустим, нулевая гипотеза звучит так — «средняя выручка с\
#пользователя не изменилась»,\
#а альтернативная — «средняя выручка с пользователя изменилась».\
#Средняя выручка с пользователя на самом деле изменилась. По\
#результатам эксперимента приняли решение отклонить нулевую\
#гипотезу. Какая ошибка совершена?\
#1 балл\
#Ошибка I рода\
#Ошибка II рода\
#Ошибок нет, всё верно"

#        text = "Вы знаете, я готов вам отдать все что есть на складе, даже если это \
#зарезервировано под других клиентов – думаю я найду нужные слова для \
#своего руководства, ведь вы очень важный для нас заказчик. \
#Какая это стратегия разрешения разногласий?\
#избегание\
#компромисс\
#приспособление\
#соперничество\
#сотрудничество"

#
#        response = giga.chat("Перед тобой распознанный текст со скриншота экрана монитора. Необходимо ответить на экзаменационный вопрос представленный на экране. Возможно, в запросе кроме самого экзаменационного задания есть паразитные слова и символы. Вопрос задан на русском языке. Возможно небольшое количество английских символов и терминов. В ответе укажи только номер правильного ответа. Если это первый по очереди ответ или ответ А, выведи 1. Если второй или В выведи 2, если С - 3, D - 4, E - 5, F - 6. Не давай никаких комментариев. Если по условиям задачи необходимо указать более одного ответа, выведи все правильные варианты через пробел. В ответах не может быть более шести вариантов ответов. Если не удаётся найти ответ, выведи 0. Если получился вариант ответа больше 6 выведи 0. Далее идёт распознанный текст задания:" + text)

#        text = "Сбер Мини-МВА 16 поток. Экзамен. Коммуникации и влияние.\
#Что такое Сила стратегии?\
#о комбинация интереса и зависимости сторон (НАОС – наилучшая альтернатива обсуждаемому соглашению)\
#о (от лат. coalitio — союз) — добровольное объединение сторон для достижения своих целей\
#о создание плана действий (переговорной карты), который лишит другую сторону возможности маневра и вынудит согласиться с вами\
#о проактивность, инициатива, определяющие быстроту воздействия на визави, нацеленности на результат"

#Что такое гибкий диалог?\
#обоснование своих требований для максимизации выгодности соглашения\
#нахождение взаимовыгодных условий для создания прочного и реализуемого соглашения\
#получение обязательств для обеспечения исполнения соглашения"

#Что такое переговоры? Выбери наиболее точный ответ из предложенных вариантов:\
#процесс обсуждения между сторонами для достижения соглашения о сделке, условиях контракта или разрешения конфликта интересов.\
#досудебное обсуждение между сторонами для достижения мирового соглашения и избежания судебного разбирательства.\
#тонкий инструмент урегулирования международных споров или заключения договоров между государствами без применения силы.\
#процесс разрешения разногласий с целью получения обязательств от визави если есть взаимный интерес к сделке и право вето.\
#коммуникация между людьми с целью нахождения взаимоприемлемого решения по общим вопросам или разногласиям."

        response = giga.chat(
    {
        "function_call": "auto",
        "messages": [
            {
                "role": "user",
                "content": "Ты - студент, сдающий экзамен. Перед тобой текст, распознанный со скриншота экрана монитора. \
Необходимо ответить на экзаменационное задание, описанное в тексте (входные данные). \
Возможно, в тексте кроме самого экзаменационного задания есть паразитные слова и символы, которые надо проигнорировать. \
Экзаменационное задание сформулировано на русском языке. В тексте может быть небольшое количество английских символов, формул и терминов. \
В самом задании (во входных данных) могут быть указаны следующие данные: \
- Описание задачи; \
- Несколько вариантов ответов, каждый из которых начинается с новой строки или пронумерован 1, 2, 3, 4, 5 и т.д. или A, B, C, D и т.д. \
Формат ответа: В ответе должна быть представлена валидная строка JSON с двойными кавычками. \
Сформируй JSON строку, которая содержит номер правильного ответа и текстовое описание этого варианта. \
В случае нескольких правильных ответов, необходимо сформировать строку с номерами ответов и их текстовыми описаниями. \
Например: Если это первый (верхний) по очереди ответ или ответ А, выведи {\"1\":\"описание первого ответа\"}. \
Если второй или В выведи {\"2\":\"описание второго ответа\"}, и т.д. \
Не давай никаких дополнительных комментариев. Если по условиям задачи необходимо указать более одного ответа (множественный выбор), выведи все правильные варианты. \
Например: {\"1\":\"описание первого ответа\",\"2\":\"описание второго ответа\"} \
Ограничения: \
- В ответах не может быть более 8 восьми вариантов. Если ты распознал больше 8 восьми вариантов ответа, выведи {\"0\":\"Ошибка\"} \
- Если в задании не указано, что требуется выбрать более одного варианта ответа, всегда выбирай только один правильный. \
- Если в задании точно указано количество верных вариантов, выбирай именно такое количество правильных ответов. \
- Если текст задания пустой, выведи {\"0\":\"Ошибка\"} \
- Если не удаётся найти ответ, выведи {\"0\":\"Ошибка\"} \
- Если задание не сформулировано или распознано не полностью, выведи {\"0\":\"Ошибка\"} \
- Если получился вариант ответа больше 8 восьми выведи {\"0\":\"Ошибка\"} \
Далее идёт распознанный текст экзаменационного задания:" + text,
                "attachments": ["3893a119-2c27-43c2-9ec6-f61525f3c12e"],
#                "attachments": ["fb12af8c-2f5d-43a2-9948-7b0067199dae"],
#                "attachments": ["3afc7ace-ce17-4316-80f3-f0cf655fe3da"],
            }
        ],
#        "temperature": 0.1
    })


        try:
            response_content = response.choices[0].message.content
            response_json = json.loads(response_content)
            return response_json
        except json.JSONDecodeError as e:
            self.log_message(f"Ошибка парсинга JSON: {e}")
            self.log_message(f"Полученный текст: {response_content}")
            return f"Ошибка при распознавании текста от Гигачат"
        except Exception as e:
            self.log_message(f"Другая ошибка: {e}")
            return f"Ошибка при обработке ответа"

    def extract_text_with_moondream2(self, image):
        """
        Извлечение текста с помощью Moondream2
        """
        try:
            # Кодирование изображения
            image_embeds = self.model.encode_image(image)

            # Запрос для точного извлечения текста
            question = "Read all text visible in this image exactly as it appears. Copy the text verbatim without interpretation, explanation, or addition. Preserve exact spelling, numbers, and formatting."

            # Генерация ответа
            answer = self.model.answer_question(
                image_embeds=image_embeds,
                question=question,
                tokenizer=self.tokenizer
            )

            return answer.strip()

        except Exception as e:
            return f"Ошибка при распознавании текста: {str(e)}"

#    def send_notifications_sync(self, numbers, **kwargs):
    def send_notifications_sync(self, answers, **kwargs):
        """
        Синхронный метод для отправки уведомлений
        """
        async def async_wrapper():
            async with TelegramClient(self.session_name, self.api_id, self.api_hash) as client:
                self.client = client
#                await self._send_notifications_async(numbers, **kwargs)
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

#    async def _send_notifications_async(self, numbers, recipient='LinuxGodsWorkaholicBot', 
#                                      delay_between_messages=5, delay_between_numbers=30):
#        """
#        Асинхронная реализация отправки уведомлений
#        """
#        try:
#            entity = await self.client.get_entity(recipient)
#            
#            for i, count in enumerate(numbers):
#                self.log_message(f"Отправка числа {count}, {count} раз(а)")
#                
#                for message_num in range(count):
#                    await self.client.send_message(entity, str(count))
#                    self.log_message(f"Отправлено сообщение {message_num + 1}/{count}")
#                    
#                    if message_num < count - 1:
#                        await asyncio.sleep(delay_between_messages)
#                
#                if i < len(numbers) - 1:
#                    self.log_message(f"Ожидание {delay_between_numbers} сек...")
#                    await asyncio.sleep(delay_between_numbers)
#            
#            self.log_message("Все уведомления отправлены!")
#            
#        except Exception as e:
#            self.log_message(f"Ошибка: {e}")
#
#
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
#                current_frame_rotated = cv2.rotate(current_frame_captured, cv2.ROTATE_180)
                current_frame_rotated = cv2.flip(current_frame_captured, -1)

                # Обрезаем до 1280x720 если они больше
#                current_frame = self.center_crop(current_frame_rotated, 1280, 720)
                current_frame = self.center_crop(current_frame_rotated, 1366, 768)
#                current_frame = self.center_crop(current_frame_rotated, 1600, 900)

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
                        time.sleep(20 - (time.time() - start_time))
                        continue

                # Обработка изображения
                print("Обнаружены значительные изменения, обрабатываем изображение...")

                # Сохранение оригинального изображения
                orig_image = self.save_image(current_frame, "original")
#                orig_image = "images/Screenshot 2025-10-23 09-58-38.png"

#                processed_image = self.preprocess_image(current_frame)

                # Распознавание текста
                self.log_message("Распознавание текста...")
#                text = self.extract_text_with_moondream2(processed_image)
                text = self.extract_text_with_yandex(orig_image)

                # Вывод результата
                print("\n" + "="*50)
                print("РАСПОЗНАННЫЙ ТЕКСТ:")
                print("="*50)
                print(text)
                print("="*50)

# DUMMY
#                self.log_message("\nЗапрос модели...")
#                answer = self.query_gigachat(text)
#
#                # Вывод результата
#                print("\n" + "="*50)
#                print("ОТВЕТ МОДЕЛИ:")
#                print("="*50)
#                print(answer)
#                print("="*50)

                answer = self.query_gigachat_rag(text)

                # Вывод результата
                print("\n" + "="*50)
                print("ОТВЕТ МОДЕЛИ RAG:")
                print("="*50)
                print(answer)
                print("="*50)


#                numbers = self.string_to_int_array(answer)
#                numbers = [int(key) for key in answer.keys()]
#                continue

                # Отправляем уведомления
                self.log_message("\nОтправка уведомлений...")
                self.send_notifications_sync(
#                    numbers=numbers,
                    answers=answer,
                    recipient='LinuxGodsWorkaholicBot',
                    delay_between_messages=4,
                    delay_between_numbers=10
                )

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
    CAMERA_INDEX = 1  # 0 - обычно встроенная камеры, 1 - внешняя камера
    SIMILARITY_THRESHOLD = 0.99  # 95% схожести

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
