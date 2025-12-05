import getpass
import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

if "GIGACHAT_CREDENTIALS" not in os.environ:
    os.environ["GIGACHAT_CREDENTIALS"] = getpass.getpass("Введите ключ авторизации GigaChat API: ")

from langchain_gigachat.chat_models import GigaChat

llm = GigaChat(verify_ssl_certs=False)

from langchain.schema import HumanMessage

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
)

# Загрузка документов
#loader = TextLoader("problems/problem_p.txt")
#loader = TextLoader("problems/problem_p1m.txt")
#loader = TextLoader("UX/slides_m.txt")
loader = TextLoader("finance/slides_pres_m.txt")
documents = loader.load()

# 2. Specify the headers to split on
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

# 3. Initialize the MarkdownHeaderTextSplitter
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

docs = []
for doc in documents:
    split_docs = markdown_splitter.split_text(doc.page_content)
    docs.extend(split_docs)

documents = docs

print(f"Total documents: {len(documents)}")

from chromadb.config import Settings
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings
from langchain_chroma import Chroma

embeddings = GigaChatEmbeddings(
    credentials=os.environ["GIGACHAT_CREDENTIALS"], verify_ssl_certs=False
)

# Указываем путь для сохранения
#persist_directory = "./chroma_db_p_pres"
#persist_directory = "./chroma_db_u_pres"
persist_directory = "./chroma_db_f_pres"
db = Chroma.from_documents(
    documents,
    embeddings,
    client_settings=Settings(anonymized_telemetry=False),
    persist_directory=persist_directory  # Добавляем путь для сохранения
)

# Явно сохраняем базу
#db.persist()
print("База данных автоматически сохранена в:", persist_directory)

