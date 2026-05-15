from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import base64
import json
from pydantic import BaseModel
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pymupdf4llm
import fitz
import tempfile
import os

class Chunk(BaseModel):
  content: str
  subject: str | None = None
  chapter: float | None = None
  lesson: float | None = None
  topic: str | None = None

class ChunksQualityRequest(BaseModel):
  chunks: list[Chunk]
  query: str

app = FastAPI()
# Dowload specific dictionary and translate human words to number
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')

# Download model
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
model.eval()

@app.get("/")
def read_root():
    return {"Hello": "Testmaker App"}

@app.post("/pdf-previews")
async def get_pdf_previews(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        doc = fitz.open(temp_path)
        previews = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=50)
            img_base64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
            previews.append({
                "page": page_num + 1,
                "thumbnail": f"data:image/png;base64,{img_base64}"
            })
        doc.close()
        os.unlink(temp_path)
        return {"numPages": len(previews), "previews": previews}
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Failed to generate previews: {str(e)}")

@app.post("/extract-pdf")
async def extract_pdf(file: UploadFile = File(...), pages: str = Form(None)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save the uploaded file to a temporary location
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        page_list = None
        if pages:
            try:
                selected_pages = json.loads(pages)
                page_list = [p - 1 for p in selected_pages]
            except Exception:
                pass

        # Extract Markdown using pymupdf4llm
        md_text = pymupdf4llm.to_markdown(temp_path, pages=page_list, ignore_graphics=True)

        # Get number of pages using fitz (PyMuPDF)
        doc = fitz.open(temp_path)
        num_pages = len(doc)
        doc.close()

        # Clean up temp file
        os.unlink(temp_path)

        return {
            "text": md_text,
            "numPages": num_pages
        }
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@app.post("/chunks-rerank")
def chunks_quality(request: ChunksQualityRequest):
  pairs = [[request.query, chunk.content] for chunk in request.chunks]
  with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    raw_scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    scores_list = raw_scores.tolist()
  results = []
  for i in range(len(scores_list)):
    results.append({
      "content": request.chunks[i].content,
      "score": float(scores_list[i])
    })
  results.sort(key=lambda x: x["score"], reverse=True)
  return {"results": results}