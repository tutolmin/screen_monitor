import getpass
import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

if "GIGACHAT_CREDENTIALS" not in os.environ:
    os.environ["GIGACHAT_CREDENTIALS"] = getpass.getpass("Введите ключ авторизации GigaChat API: ")

from langchain_gigachat.chat_models import GigaChat

llm = GigaChat(verify_ssl_certs=False)

from langchain.schema import HumanMessage

question = """Каких типов бывает переговорная КАРТА?"""
#question = """Ты - студент, сдающий экзамен. Перед тобой текст, распознанный со скриншота экрана монитора. 
#Необходимо ответить на экзаменационный вопрос, описанный в тексте (входные данные). 
#Возможно, в тексте кроме самого экзаменационного задания есть паразитные слова и символы, которые надо проигнорировать. 
#Экзаменационное задание сформулировано на русском языке. В тексте может быть небольшое количество английских символов, формул и терминов. 
#В самом задании (во входных данных) могут быть указаны следующие данные: 
#- Заголовок: СБЕР УНИВЕРСИТЕТ Сбер Мини-МВА 16 поток. Экзамен; 
#- Описание задачи; 
#- Несколько вариантов ответов, каждый с новой строки или пронумерованные 1, 2, 3, 4, 5, 6, 7 и т.д. или A, B, C, D и т.д. 
#Формат ответа: В ответе должен быть представлен валидная строка JSON с двойными кавычками. 
#Сформируй JSON строку, которая содержит номер правильного ответа и текстовое описание этого варианта. 
#В случае нескольких правильных ответов, необходимо сформировать строку с номерами ответов и их текстовыми описаниями. 
#Например: Если это первый (верхний) по очереди ответ или ответ А, выведи {"1":"описание первого ответа"}. 
#Если второй или В выведи {"2":"описание второго ответа"}, и т.д. 
#Не давай никаких дополнительных комментариев. Если по условиям задачи необходимо указать более одного ответа, выведи номера всех правильных вариантов. 
#Далее идёт распознанный текст экзаменационного задания: 
#СБЕР 
#УНИВЕРСИТЕТ 
#Сбер Мини-МВА 16 поток Экзамен по дисципине "Коммуњ 
#Переговорная карта может быть (множественный выбор): 
#Тактическая 
#B 
#Пестрая 
#Наивная 
#Комплексная"""

#Например: {"1":"описание первого ответа","2":"описание второго ответа"} 
#Ограничения: 
#- В ответах не может быть более семи вариантов. Если ты распознал больше чем 7 вариантов ответа, выведи {"0":"Ошибка"} 
#- Если в задании не указано, что требуется выбрать более одного варианта ответа, всегда выбирай только один правильный. 
#- Если текст задания пустой, выведи {"0":"Ошибка"} 
#- Если не удаётся найти ответ, выведи {"0":"Ошибка"} 
#- Если задание не сформулировано или распознано не полностью, выведи {"0":"Ошибка"} 
#- Если получился вариант ответа больше 7 выведи {"0":"Ошибка"}

#response = llm([HumanMessage(content=question)]).content[0:200]
#print(response)

#response = llm.invoke([HumanMessage(content=question)])
#print(response.content[0:200])

from langchain_community.document_loaders import TextLoader
#from langchain.document_loaders import TextLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)

#loader = TextLoader("structured_presentation_clean.txt")
#loader = TextLoader("leadership/structured_presentation.txt")
#loader = TextLoader("leadership/w_clean.txt")
#loader = TextLoader("energy/everything.txt")
#loader = TextLoader("energy/structured_presentation.txt")
#loader = TextLoader("analytics/structured_presentation.txt")
#loader = TextLoader("problems/problem_w.txt")
#loader = TextLoader("UX/uxw.txt")
loader = TextLoader("finance/fw.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
#   Presentation
#    chunk_size=400,
#    chunk_overlap=100,
#   Webinar
    chunk_size=1000,
    chunk_overlap=200,
)
documents = text_splitter.split_documents(documents)
print(f"Total documents: {len(documents)}")

from chromadb.config import Settings
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings
from langchain_chroma import Chroma

embeddings = GigaChatEmbeddings(
    credentials=os.environ["GIGACHAT_CREDENTIALS"], verify_ssl_certs=False
)

# Указываем путь для сохранения
#persist_directory = "./chroma_db_c_pres"
#persist_directory = "./chroma_db_l_web"
#persist_directory = "./chroma_db_e_web"
#persist_directory = "./chroma_db_l_pres"
#persist_directory = "./chroma_db_e_pres"
#persist_directory = "./chroma_db_a_pres"
#persist_directory = "./chroma_db_p_web"
#persist_directory = "./chroma_db_u_web"
persist_directory = "./chroma_db_f_web"
db = Chroma.from_documents(
    documents,
    embeddings,
    client_settings=Settings(anonymized_telemetry=False),
    persist_directory=persist_directory  # Добавляем путь для сохранения
)

# Явно сохраняем базу
#db.persist()
print("База данных автоматически сохранена в:", persist_directory)


#docs = db.similarity_search(question, k=3)
#print(len(docs))
#
#print(f"... {str(docs[0])[0:300]} ...")
#print(f"... {str(docs[1])[0:300]} ...")
#print(f"... {str(docs[2])[0:300]} ...")
#
#from langchain.chains import RetrievalQA
#
#qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
#
#result = qa_chain({"query": question})
#
#print(result)
#
