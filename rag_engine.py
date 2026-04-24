import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()
def load_and_process_pdf(pdf_path):
    pdf=PdfReader(pdf_path)
    text=""
    for lines in pdf.pages:
        page_text = lines.extract_text()
        page_text = page_text.encode("utf-8", errors="ignore").decode("utf-8")
        text += page_text
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks=text_splitter.split_text(text)
    return chunks

def vectorize(chunks):
    embedding_model=OllamaEmbeddings(
        model=' nomic-embed-text:v1.5'
    )
    vector_storage=Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory='./chroma_db'
    )
    return vector_storage
# if __name__ == '__main__':
#     pdf_path='example.pdf'
#     text=load_and_process_pdf(pdf_path)
#     print(len(text))
def find_similarity(query,vector_storage,k=4):
    similar=vector_storage.similarity_search(query,k)
    result=""
    for chunk in similar:
        result+=chunk.page_content
    return result
def get_answer(query,retrived):

    prompt=f"""
    You are a helpful assistant. 
    Answer the question using only the context below.
    If the answer is not in the context, say that the answer is not in the context gracefully
    Context: {retrived}
    Question: {query}"""
    model=ChatOllama(
        model='phi3:mini',temperature=0)
    response=model.invoke(prompt)
    return response.content
# if __name__ == '__main__':
#     pdf_path = 'example.pdf'
#     chunks = load_and_process_pdf(pdf_path)
#     vectorstore = vectorize(chunks)
#     query=input("Enter query: ")
#     k=int(input("Enter k: "))
#     similar=find_similarity(query,vectorstore,k)
#     # print(similar)
#     # # print(type(similar[0].page_content))
#     answer=get_answer(query,similar)
#     print(answer)