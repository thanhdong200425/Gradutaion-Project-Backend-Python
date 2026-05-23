import base64
import json
import os
import tempfile

import fitz
import pymupdf4llm
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from classifier import classifier


class Chunk(BaseModel):
    content: str
    subject: str | None = None
    chapter: float | None = None
    lesson: float | None = None
    topic: str | None = None


class ChunksQualityRequest(BaseModel):
    chunks: list[Chunk]
    query: str


class Question(BaseModel):
    id: str
    content: str


class DifficultyRequest(BaseModel):
    questions: list[Question]


app = FastAPI()

# Scanned/image pages rarely have selectable text; below this avg → Paddle OCR.
MIN_NATIVE_CHARS_PER_PAGE = 10

_paddle_ocr = None


def _get_paddle_ocr():
    global _paddle_ocr
    if _paddle_ocr is None:
        from paddleocr import PaddleOCR

        # PP-OCRv5 mobile (~21 MB) — much lighter than PaddleOCR-VL (~1.9 GB).
        # latin rec model covers Vietnamese (vi is a Latin-script language).
        _paddle_ocr = PaddleOCR(
            device="cpu",
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    return _paddle_ocr


def _ocr_result_page_text(res) -> str:
    texts = res.get("rec_texts") or []
    return "\n".join(t.strip() for t in texts if t and str(t).strip())


def ocr_pdf_with_paddle(pdf_path: str, page_list: list[int] | None) -> str:
    ocr = _get_paddle_ocr()
    pages_result = list(ocr.predict(pdf_path))

    if page_list is not None:
        page_set = set(page_list)
        pages_result = [r for r in pages_result if r and r.get("page_index") in page_set]

    parts = [_ocr_result_page_text(r) for r in pages_result]
    parts = [p for p in parts if p]
    return "\n\n".join(parts)


def _page_indices(doc: fitz.Document, page_list: list[int] | None) -> list[int]:
    if page_list is not None:
        return [i for i in page_list if 0 <= i < len(doc)]
    return list(range(len(doc)))


def pdf_needs_ocr(doc: fitz.Document, page_list: list[int] | None) -> bool:
    indices = _page_indices(doc, page_list)
    if not indices:
        return True
    total_chars = sum(len(doc[i].get_text("text").strip()) for i in indices)
    return (total_chars / len(indices)) < MIN_NATIVE_CHARS_PER_PAGE


tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")
model.eval()


@app.get("/")
def read_root():
    return {"Hello": "Testmaker App"}


@app.post("/chunks-rerank")
def chunks_quality(request: ChunksQualityRequest):
    pairs = [[request.query, chunk.content] for chunk in request.chunks]
    with torch.no_grad():
        inputs = tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        raw_scores = (
            model(**inputs, return_dict=True)
            .logits.view(
                -1,
            )
            .float()
        )
        scores_list = raw_scores.tolist()
    results = []
    for i in range(len(scores_list)):
        results.append(
            {"content": request.chunks[i].content, "score": float(scores_list[i])}
        )
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"results": results}


@app.post("/predict/difficulty")
def predict_difficulty(request: DifficultyRequest):
    contents = [q.content for q in request.questions]
    predictions = classifier.predict(contents)
    results = [
        {"id": request.questions[i].id, **predictions[i]}
        for i in range(len(predictions))
    ]
    return {"results": results}


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
            previews.append(
                {
                    "page": page_num + 1,
                    "thumbnail": f"data:image/png;base64,{img_base64}",
                }
            )
        doc.close()
        os.unlink(temp_path)
        return {"numPages": len(previews), "previews": previews}
    except Exception as e:
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(
            status_code=500, detail=f"Failed to generate previews: {str(e)}"
        )


@app.post("/extract-pdf")
async def extract_pdf(file: UploadFile = File(...), pages: str = Form(None)):
    filename = file.filename
    if not filename or not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        suffix = os.path.splitext(filename)[1]
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

        doc = fitz.open(temp_path)
        num_pages = len(doc)
        use_ocr = pdf_needs_ocr(doc, page_list)
        doc.close()

        if use_ocr:
            md_text = ocr_pdf_with_paddle(temp_path, page_list)
        else:
            md_text = pymupdf4llm.to_markdown(
                temp_path, pages=page_list, ignore_graphics=True
            )
            if not md_text.strip():
                md_text = ocr_pdf_with_paddle(temp_path, page_list)

        os.unlink(temp_path)

        return {"text": md_text, "numPages": num_pages}
    except Exception as e:
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
