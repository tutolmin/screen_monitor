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

        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
#        self.client = TelegramClient(session_name, api_id, api_hash)

        self.narrative = ""
        self.search_query = ""
        self.question_type = "Multiple Choice"

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
        #        "content": encode_file("images/8930.jpg")}

        url = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"

        token = ""

        headers= {"Content-Type": "application/json",
                  "Authorization": "Bearer {:s}".format(token),
                  "x-folder-id": "b1ghg3qttqeg3e6qpgp5",
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
#            print("Введите текст кейса (Ctrl+D или Ctrl+Z для завершения):")
#            lines = []
#            while True:
#                try:
#                    line = input()
#                except EOFError:
#                    break
#                lines.append(line)
#            response_json["result"]["textAnnotation"]["fullText"] = '\n'.join(lines)

            # clear headers
#            pattern = r'^СБЕР\n|^УНИВЕРСИТЕТ\n|^НАЗАД\n|^ПРОПУСТИТЬ.*\n|^ЗАВЕРШИТЬ\n|Пройден.*\n|^Сбер Мини-МВА.*\n|^Прокторинг.*\n'
            pattern = r'^СБЕР\n|.*ЕРСИТЕТ\n|^НАЗАД\n|^ПРОПУСТИТЬ.*\n|^ОТВЕТИТЬ.*\n|^ЗАВЕРШИТЬ.*\n|Пройден.*\n|.*Мини-МВА.*\n|^Прокторинг.*\n|^Тестовая.*\n|^Кейс.*\n'
#            pattern = r'^СБЕР\n|.*ЕРСИТЕТ\n|^НАЗАД\n|^ПРОПУСТИТЬ.*\n|^ОТВЕТИТЬ.*\n|^ЗАВЕРШИТЬ.*\n|Пройден.*\n|.*Мини-МВА.*\n|^Прокторинг.*\n|^Тестовая.*\n'
            text_clear = re.sub(pattern, '', response_json["result"]["textAnnotation"]["fullText"], 
                                flags=re.IGNORECASE | re.MULTILINE)

            return text_clear[:1500]
#            return text_clear[:1600]
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

        from langchain_gigachat.chat_models import GigaChat

        giga = GigaChat(
            credentials=os.environ["GIGACHAT_CREDENTIALS"],
            model="GigaChat-Max",
#            model="GigaChat",
            verify_ssl_certs=False,
            timeout=30,
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

        persist_directory = "./chroma_db_f_web"

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

        persist_directory_pres = "./chroma_db_f_pres"

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
            search_type="mmr",
            search_kwargs={
#                "k": 5,           # Финальное количество чанков
#                "fetch_k": 20,    # Сколько кандидатов рассматривать initially
#                "lambda_mult": 0.8  # Коэффициент разнообразия (0-1), где 1 - максимальное разнообразие
                "k": 3,           # Финальное количество чанков
                "fetch_k": 10,    # Сколько кандидатов рассматривать initially
                "lambda_mult": 0.4  # Коэффициент разнообразия (0-1), где 1 - максимальное разнообразие
            }
        )
        retriever_pres = db_pres.as_retriever(
#            search_kwargs={"k": 5}  # использовать 3 чанка
            search_type="mmr",
            search_kwargs={
                "k": 3,           # Финальное количество чанков
                "fetch_k": 15,    # Сколько кандидатов рассматривать initially
                "lambda_mult": 0.7  # Коэффициент разнообразия (0-1), где 1 - максимальное разнообразие
            }
#            search_kwargs={"k": 6}  # использовать 3 чанка
        )
        from langchain.retrievers import EnsembleRetriever
        # Создаем ансамблевый ретривер
        ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever, retriever_pres],
            weights=[0.4, 0.6]  # веса для каждого ретривера
        )

        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
        from langchain_core.prompts import ChatPromptTemplate
        from langchain.callbacks.tracers import ConsoleCallbackHandler

#Для ответа на вопрос необходимо провести анализ, выполнить вычисления или построить логическую цепочку.
        template = """Ты - студент, сдающий экзамен.
Дисциплина Финансы в новой экономике.
Перед тобой текст, распознанный со скриншота экрана монитора.
Необходимо ответить на экзаменационное задание, описанное в тексте (входные данные).
Для ответа на вопрос необходимо провести анализ возможных вариантов ответа.
Опирайся в первую очередь на контекст, поскольку он содержит выдержки из материалов курса по дисциплине.

Экзаменационное задание сформулировано на русском языке. 
В тексте может быть небольшое количество английских символов, формул и терминов.
При формулировании ответа оцени все предложенные варианты ответа.

Формат ответа: Суммаризация рассуждений объёмом не более 5000 символов.

Контекст: {context}

Экзаменационное задание: {input}"""

        prompt = ChatPromptTemplate.from_template(template)
        question_answer_chain = create_stuff_documents_chain(giga, prompt)

#        print(self.search_query)

#        if(len(self.narrative)>0):
#
#            # Промпт для извлечения поискового запроса
#            unification_prompt = ChatPromptTemplate.from_template("""
#Перед тобой несколько фрагментов одного текста. 
#В отдельных фрагментах этого текста могут присутствовать части других фрагментов. 
#Твоя задача восстановить целостность путем удаления дубликатов.
#
#Формат ответа: Очищенный от дубликатов и объединенный текст.
#
#Фрагменты текста:
#
#Далее идёт распознанный текст, который необходимо обработать: {input} """)
#
#            # Цепочка для извлечения запроса
#            unification_chain = unification_prompt | giga
#
#            def unify_with_llm(full_input):
#                """Использует LLM для извлечения поискового запроса"""
#                text = full_input["input"]
#                search_query = ""
#                try:
#                    search_query = unification_chain.invoke({"input": text}, config={"callbacks": [ConsoleCallbackHandler()]}).content
#                except Exception as e:
#                    self.log_message(f"Ошибка запроса к LLM: {e}")
#                print(f"🔍 Объединённый нарратив: '{search_query}'")
#                return search_query
#
#            # Шаг 1: Извлекаем search_query
#            self.narrative = unify_with_llm({"input": self.narrative})
#
        from langchain_core.documents import Document  # Добавляем импорт

        # Функция для преобразования строки в список документов
        def prepare_context(x):
            if self.narrative and self.narrative.strip():
                # Если есть narrative, преобразуем его в список документов
                return [Document(page_content=self.narrative)]
            else:
                # Иначе используем retriever
                return ensemble_retriever.invoke(self.search_query)

        context_selector = RunnableLambda(prepare_context)

        # Функция для извлечения текста из input
        def extract_input_text(data):
            if isinstance(data, dict) and 'input' in data:
                return data['input']
            return data

        rag_chain = (
            {
#                "input": RunnablePassthrough(extract_input_text),  # Используем RunnablePassthrough для input
                "input": RunnableLambda(extract_input_text),  # Извлекаем текст из input
                "context": context_selector
            }
            | question_answer_chain
        )

#
#        # Определяем контекст в зависимости от наличия narrative
#        context_selector = RunnableLambda(
#            lambda x: self.narrative if self.narrative and self.narrative.strip()
#            else ensemble_retriever.invoke(self.search_query)
#        )
#
#        rag_chain = (
#            {
#                "input": lambda x: text,
#                "context": context_selector
#            }
#            | question_answer_chain
#        )

#        print(self.search_query)
#        rag_chain = (
#            {
#                "input": lambda x: x["input"],
#                "context": lambda x: self.narrative,
##                "context": RunnableLambda(lambda x: self.search_query) | ensemble_retriever
##                "context": RunnableLambda(lambda x: self.search_query) | retriever_pres
#            }
#            | question_answer_chain
#        )
#
        result = ""
        try:
        # Вариант 3: Универсальный
            result = rag_chain.invoke({"input": text}, config={"callbacks": [ConsoleCallbackHandler()]})
#            result = rag_chain.invoke({"input": text})
        except Exception as e:
            self.log_message(f"Ошибка запроса к LLM rag: {e}")
        reason = result if not isinstance(result, dict) else result.get("output", result)


        # Промпт для классификации задания - должен быть ChatPromptTemplate
        reason_template = """Ты - студент, сдающий экзамен.
Дисциплина Финансы в новой экономике.
Перед тобой текст, распознанный со скриншота экрана монитора.
Возможно, в тексте кроме самого экзаменационного задания есть паразитные слова и символы, которые надо проигнорировать.
Возможно, в тексте есть название кнопок навигации по тесту, такие как Продолжить, Назад, Ответить, Завершить.

Экзаменационное задание сформулировано на русском языке. В тексте может быть небольшое количество английских символов, формул и терминов.
В самом задании (во входных данных) указаны следующие данные: 
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

Если нет специального уточнения о множественном, выбери один правильный ответ.
Если указано точное количество ответов, которые необходимо указать, выбери именно такое число вариантов.
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


        if self.question_type == "Numeric/Short Answer":

            # Промпт для классификации задания - должен быть ChatPromptTemplate
            reason_template = """Ты - студент, сдающий экзамен.
Дисциплина Финансы в новой экономике.
Перед тобой текст, распознанный со скриншота экрана монитора.
Возможно, в тексте кроме самого экзаменационного задания есть паразитные слова и символы, которые надо проигнорировать.
Возможно, в тексте есть название кнопок навигации по тесту, такие как Продолжить, Назад, Ответить, Завершить.

Экзаменационное задание сформулировано на русском языке. В тексте может быть небольшое количество английских символов, формул и терминов.
В самом задании (во входных данных) указано описание задачи, которую необходимо решить.

В запросе также приведены твои собственные рассуждения по этой задаче.
Необходимо ответить на экзаменационное задание. 

Формат ответа: В ответе должна быть представлена валидная строка JSON с двойными кавычками следующего вида {{"1":"описание ответа"}}.
В JSON строке ключ - это всегда цифра 1.
Описание ответа должно содержать следующюю информацию: число - итоговое значение ответа без каких либо дополнительных комментариев. 
Далее в описании ответа через точку с запятой должна быть указана последовательность математических действий, которая приводит к ответу.
Не добавляй никаких текстовых пояснений. Требуется привести только расчет.
Используй сокращения тыс., млн., млрд., чтобы не выводить большое количество нулей.

Ограничения:
- Ключ варианта ответа всегда цифра 1
- Если текст задания пустой, выведи {{"0":"Ошибка"}}
- Если не удаётся найти ответ, выведи {{"0":"Ошибка"}}
- Если задание не сформулировано или распознано не полностью, выведи {{"0":"Ошибка"}}

Не давай никаких дополнительных комментариев. 

Экзаменационное задание: {input}

Твои размышления по этому вопросу: {reason}"""



        prompt = ChatPromptTemplate.from_template(reason_template)

        # Создаем цепочку для классификации
        reasoning_chain = prompt | giga

        result = ""
        try:
            # Шаг 2: Классифицируем используя ТОЛЬКО search_query
            result = reasoning_chain.invoke({"reason": reason, "input": text}, config={"callbacks": [ConsoleCallbackHandler()]})
#            result = reasoning_chain.invoke({"reason": reason, "input": text})
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
        from langchain_gigachat.chat_models import GigaChat
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
        from langchain_core.prompts import ChatPromptTemplate
        from langchain.callbacks.tracers import ConsoleCallbackHandler

        giga = GigaChat(
            credentials=os.environ["GIGACHAT_CREDENTIALS"],
            model="GigaChat-Max",
            verify_ssl_certs=False,
            timeout=30,
        )

        # Промпт для извлечения поискового запроса
        query_extraction_prompt = ChatPromptTemplate.from_template("""
Перед тобой текст, распознанный со скриншота (снимка экрана).
Текст содержит экзаменационное задание и, возможно, несколько вариантов ответа.
Дисциплина Финансы в новой экономике.
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
                search_query = query_extraction_chain.invoke({"input": text}, config={"callbacks": [ConsoleCallbackHandler()]}).content
#                search_query = query_extraction_chain.invoke({"input": text}).content
            except Exception as e:
                self.log_message(f"Ошибка запроса к LLM: {e}")
            print(f"🔍 Извлеченный поисковый запрос: '{search_query}'")
            return search_query

        self.search_query = extract_with_llm({"input": text})

        # Промпт для классификации задания - должен быть ChatPromptTemplate
        classification_prompt = ChatPromptTemplate.from_template(""" 
Ты — эксперт по анализу заданий для экзамена по дисциплине «Финансы в новой экономике». Определи тип задания строго по его структуре и формату ответа:

* Multiple Choice — если задание содержит вопрос и перечень вариантов ответа (обычно от 3 до 5), из которых нужно выбрать один или несколько. В тексте явно присутствуют формулировки вроде: «Выберите правильный вариант», «Какие утверждения верны?», «A) … B) … C) …». Ответ выбирается из предложенного списка.

* Numeric/Short Answer — если задание требует выполнить расчёт, анализ или интерпретацию данных, и в результате нужно ввести числовое значение, процент, коэффициент, формулу или краткий ответ в специальное поле (например: «Рассчитайте NPV проекта», «Укажите значение коэффициента текущей ликвидности», «В ответе запишите сумму в млн руб.»). Варианты ответа не предлагаются.

Ответь строго одним словом:
Multiple Choice
Numeric/Short Answer

Не объясняй. Не добавляй комментариев. Только один из двух вариантов.

Текст для анализа: {search_query}""")

        # Создаем цепочку для классификации
        classification_chain = classification_prompt | giga

        # Шаг 1: Извлекаем search_query
        result = "Multiple Choice" 
        try:
            # Шаг 2: Классифицируем используя ТОЛЬКО search_query
#            result = classification_chain.invoke({"search_query": self.search_query}, config={"callbacks": [ConsoleCallbackHandler()]})
            result = classification_chain.invoke({"search_query": text}, config={"callbacks": [ConsoleCallbackHandler()]})
#            result = classification_chain.invoke({"search_query": self.search_query})
        except Exception as e:
            self.log_message(f"Ошибка запроса к LLM: {e}")

        final_result = result.content if hasattr(result, 'content') else result
        
        print(f"📊 Тип задания: {final_result}")
        return final_result

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

                if text is None:
                    # Обработка случая, когда OCR вернул None (произошла ошибка)
                    print("OCR завершился с ошибкой")
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

                self.log_message("\nЗапрос модели...")
                answer = self.query_gigachat_task_type(text)

                # Вывод результата
                print("\n" + "="*50)
                print("ОТВЕТ МОДЕЛИ RAG:")
                print("="*50)
                print(answer)
                print("="*50)
                self.question_type = answer

                # Narrative
                if False and answer == "Narrative":
                    self.narrative += "\n" + text  # с новой строки
                    print(f"Длина буфера: {len(self.narrative)} символов")
                else:
                    answer = self.query_gigachat_reason(text)
                    self.narrative = ""
                    print(f"Обнуляем буфер")

                    # Отправляем уведомления
                    self.log_message("\nОтправка уведомлений...")
# DUMMY
                    self.send_notifications_sync(
#                    numbers=numbers,
                        answers=answer,
                        recipient='LinuxGodsWorkaholicBot',
                        delay_between_messages=3,
                        delay_between_numbers=7
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
