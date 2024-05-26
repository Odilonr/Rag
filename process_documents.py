from langchain.docstore.document import Document
import csv
colums_to_embed = ['url', 'text']
columns_to_metadata = ["url","text","date"]

docs = []
with open('data/utt.csv', newline='', encoding='utf-8-sig') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for i, row in enumerate(csv_reader):
        to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
        values_to_embed = {k: row[k] for k in colums_to_embed if k in row}
        to_embed = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values_to_embed.items())
        newDoc = Document(page_content=to_embed, metadata =to_metadata)
        docs.append(newDoc)

from langchain.text_splitter import CharacterTextSplitter
splitter = CharacterTextSplitter    (separator='\n',
                                chunk_size=8000, 
                                chunk_overlap=0,
                                length_function=len,
                                )
documents = splitter.split_documents(docs)

import dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv()

CHROMA_PATH = 'chroma_data'

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", show_progress_bar=True)
db = Chroma.from_documents(documents, embeddings,persist_directory=CHROMA_PATH)