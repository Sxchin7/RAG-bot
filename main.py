from fastapi import FastAPI,File,UploadFile,HTTPException
from pydantic import BaseModel
import uuid
import tempfile
from rag_engine import load_and_process_pdf,vectorize,get_answer,find_similarity

app=FastAPI()
session={}

class ChatRequest(BaseModel):
    session_id: str
    query: str
class ChatResponse(BaseModel):
    answer: str
@app.post('/upload')
async def upload(file: UploadFile = File(...)):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name
        chunks=load_and_process_pdf(temp_path)
        vector_store=vectorize(chunks)
        session_id=str(uuid.uuid4())
        session[session_id]=vector_store
        return {"session_id": session_id}

@app.post('/chat')
async def chat(request:ChatRequest):
    if request.session_id not in session:
        raise HTTPException(status_code=400,detail="session id not exist")
    similar=find_similarity(request.query,session[request.session_id])
    answer=get_answer(request.query,similar)
    response=ChatResponse(answer=answer)
    return response