
import os
import io
import json
import base64
import hashlib
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError

from PIL import Image
import numpy as np


from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch

# --------------------------
# Config
# ---------------------------
STORAGE_DIR = os.environ.get("STORAGE_DIR", "./storage")
FILES_DIR = os.path.join(STORAGE_DIR, "files")
EMB_DIR = os.path.join(STORAGE_DIR, "embeddings")
DB_PATH = os.environ.get("DB_PATH", "sqlite:///./day_ai.db")

os.makedirs(FILES_DIR, exist_ok=True)
os.makedirs(EMB_DIR, exist_ok=True)

# ---------------------------
# DB setup (SQLAlchemy)
# ---------------------------
Base = declarative_base()
engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Entry(Base):
    __tablename__ = "entries"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=True)
    text = Column(Text, nullable=True)
    images = Column(Text, nullable=True)   # JSON list of file paths
    audio = Column(Text, nullable=True)    # JSON list of file paths
    created_at = Column(DateTime, default=datetime.utcnow)
    text_embedding_path = Column(String(512), nullable=True)
    # note: image embeddings are stored per image as separate .npy files; images list contains filenames

Base.metadata.create_all(bind=engine)

# ---------------------------
# Embedding models (load once)
# ---------------------------
# Text embedding model (small & fast)
TEXT_EMB_MODEL_NAME = "all-MiniLM-L6-v2"
text_model = SentenceTransformer(TEXT_EMB_MODEL_NAME)

# CLIP for images
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="AI - Trained On Your Day (backend)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Pydantic schemas
# ---------------------------
class CreateEntryResponse(BaseModel):
    id: int
    message: str

class SearchResultItem(BaseModel):
    id: int
    score: float
    title: Optional[str]
    text: Optional[str]
    images: Optional[List[str]]
    created_at: datetime

# ---------------------------
# Utilities
# ---------------------------
def save_upload_file(file: UploadFile, subfolder: str = "misc"):
    ext = os.path.splitext(file.filename)[1] or ""
    # use hash of content + timestamp for filename
    contents = file.file.read()
    file_hash = hashlib.sha1(contents + str(datetime.utcnow()).encode()).hexdigest()
    folder = os.path.join(FILES_DIR, subfolder)
    os.makedirs(folder, exist_ok=True)
    filename = f"{file_hash}{ext}"
    path = os.path.join(folder, filename)
    with open(path, "wb") as f:
        f.write(contents)
    # reset file pointer for safety
    try:
        file.file.seek(0)
    except Exception:
        pass
    return path

def save_numpy(arr: np.ndarray, name_hint: str):
    fname = hashlib.sha1((name_hint + str(datetime.utcnow())).encode()).hexdigest() + ".npy"
    path = os.path.join(EMB_DIR, fname)
    np.save(path, arr)
    return path

def load_numpy(path: str) -> np.ndarray:
    return np.load(path)

def normalize(vec: np.ndarray):
    v = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(v) + 1e-12
    return v / norm

def compute_text_embedding(text: str) -> np.ndarray:
    emb = text_model.encode([text], convert_to_numpy=True)[0]
    return normalize(emb)

def compute_image_embedding_from_pil(img: Image.Image) -> np.ndarray:
    # CLIP expects images and returns normalized embeddings
    inputs = clip_processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    image_features = image_features.cpu().numpy()[0]
    return normalize(image_features)

# Cosine similarity
def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))

# ---------------------------
# Endpoints
# ---------------------------
@app.post("/entries", response_model=CreateEntryResponse)
async def create_entry(
    title: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    images: Optional[List[UploadFile]] = None,
    audio: Optional[List[UploadFile]] = None
):
    """
    Create an entry. Accepts multipart form:
      - title (str)
      - text (str)
      - images[] (files)
      - audio[] (files)
    The server saves files, computes embeddings (text + image embeddings) and stores metadata.
    """
    db = SessionLocal()
    try:
        saved_images = []
        saved_audio = []
        image_embedding_paths = []  # we'll save per-image .npy

        # save images
        if images:
            for up in images:
                p = save_upload_file(up, subfolder="images")
                saved_images.append(p)

                # compute image embedding
                try:
                    with Image.open(p).convert("RGB") as img:
                        emb = compute_image_embedding_from_pil(img)
                    emb_path = save_numpy(emb, name_hint=p)
                    image_embedding_paths.append(emb_path)
                except Exception as ex:
                    # continue but note failure
                    print("Image embedding error:", ex)

        # save audio
        if audio:
            for up in audio:
                p = save_upload_file(up, subfolder="audio")
                saved_audio.append(p)

        # compute text embedding
        text_emb_path = None
        if text:
            try:
                emb = compute_text_embedding(text)
                text_emb_path = save_numpy(emb, name_hint=text[:200])
            except Exception as ex:
                print("Text embedding error:", ex)

        new_entry = Entry(
            title=title,
            text=text,
            images=json.dumps(saved_images) if saved_images else None,
            audio=json.dumps(saved_audio) if saved_audio else None,
            text_embedding_path=text_emb_path
        )
        db.add(new_entry)
        db.commit()
        db.refresh(new_entry)

        # store image embedding mapping file: entry_<id>_images.json listing paths (optional)
        if saved_images and image_embedding_paths:
            mapping = [{"image_path": p, "emb_path": e} for p, e in zip(saved_images, image_embedding_paths)]
            map_path = os.path.join(EMB_DIR, f"entry_{new_entry.id}_images.json")
            with open(map_path, "w") as fh:
                json.dump(mapping, fh)

        return {"id": new_entry.id, "message": "Entry saved"}
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/entries/{entry_id}")
def get_entry(entry_id: int):
    db = SessionLocal()
    try:
        ent = db.query(Entry).get(entry_id)
        if not ent:
            raise HTTPException(status_code=404, detail="Entry not found")
        return {
            "id": ent.id,
            "title": ent.title,
            "text": ent.text,
            "images": json.loads(ent.images) if ent.images else [],
            "audio": json.loads(ent.audio) if ent.audio else [],
            "created_at": ent.created_at.isoformat()
        }
    finally:
        db.close()

@app.get("/entries")
def list_entries(skip: int = 0, limit: int = 50):
    db = SessionLocal()
    try:
        q = db.query(Entry).order_by(Entry.created_at.desc()).offset(skip).limit(limit).all()
        out = []
        for ent in q:
            out.append({
                "id": ent.id,
                "title": ent.title,
                "text": ent.text,
                "images": json.loads(ent.images) if ent.images else [],
                "audio": json.loads(ent.audio) if ent.audio else [],
                "created_at": ent.created_at.isoformat()
            })
        return out
    finally:
        db.close()

@app.post("/search/text", response_model=List[SearchResultItem])
def search_text(query: str = Form(...), top_k: int = Form(6)):
    """
    Search stored entries by text similarity.
    """
    db = SessionLocal()
    try:
        # compute query embedding
        q_emb = compute_text_embedding(query)

        # load all entries with text_embedding_path
        rows = db.query(Entry).filter(Entry.text_embedding_path != None).all()
        scored = []
        for r in rows:
            try:
                emb = load_numpy(r.text_embedding_path)
                score = cos_sim(q_emb, emb)
                scored.append((score, r))
            except Exception:
                continue
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]
        return [
            SearchResultItem(
                id=e.id,
                score=float(score),
                title=e.title,
                text=e.text,
                images=json.loads(e.images) if e.images else [],
                created_at=e.created_at
            ) for score, e in top
        ]
    finally:
        db.close()

@app.post("/search/image", response_model=List[SearchResultItem])
async def search_image(file: UploadFile = File(...), top_k: int = Form(6)):
    """
    Upload an image and find similar stored images / entries.
    """
    # read uploaded file into PIL
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    q_emb = compute_image_embedding_from_pil(img)

    # iterate over image embedding mapping files
    results = []
    for fname in os.listdir(EMB_DIR):
        if not fname.endswith(".npy"):
            continue
        path = os.path.join(EMB_DIR, fname)
        try:
            emb = load_numpy(path)
            score = cos_sim(q_emb, emb)
            # determine which entry the image belongs to (we saved entry mapping jsons)
            # fallback: locate image files by filename matching pattern
            owner_entry_id = None
            # check entry image mapping files
            # naive approach: check any mapping file that references this emb path
            for mapf in os.listdir(EMB_DIR):
                if mapf.startswith("entry_") and mapf.endswith("_images.json"):
                    with open(os.path.join(EMB_DIR, mapf), "r") as fh:
                        mapping = json.load(fh)
                        for m in mapping:
                            if os.path.basename(m.get("emb_path","")) == os.path.basename(path):
                                # extract entry id from mapf name
                                try:
                                    owner_entry_id = int(mapf.split("_")[1])
                                except: pass
            results.append((score, path, owner_entry_id))
        except Exception:
            continue

    results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]

    # convert to SearchResultItem list with entry info
    out = []
    db = SessionLocal()
    try:
        for score, emb_path, entry_id in results:
            if entry_id is not None:
                ent = db.query(Entry).get(entry_id)
                if ent:
                    out.append(SearchResultItem(
                        id=ent.id,
                        score=float(score),
                        title=ent.title,
                        text=ent.text,
                        images=json.loads(ent.images) if ent.images else [],
                        created_at=ent.created_at
                    ))
    finally:
        db.close()
    return out

@app.get("/file/{path:path}")
def serve_file(path: str):
    # serve files from FILES_DIR (careful in prod - add auth)
    abs_path = os.path.join(FILES_DIR, path)
    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(abs_path)

@app.get("/export/all")
def export_all():
    """
    Export all entries metadata (not binary files) as JSON for backup.
    To move files between machines, also tar up ./storage/files.
    """
    db = SessionLocal()
    try:
        rows = db.query(Entry).all()
        out = []
        for r in rows:
            out.append({
                "id": r.id,
                "title": r.title,
                "text": r.text,
                "images": json.loads(r.images) if r.images else [],
                "audio": json.loads(r.audio) if r.audio else [],
                "created_at": r.created_at.isoformat(),
                "text_embedding_path": r.text_embedding_path
            })
        return JSONResponse(content=out)
    finally:
        db.close()
