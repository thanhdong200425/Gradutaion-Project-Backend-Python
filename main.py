from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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