from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
import subprocess

DATA_DIR = os.environ.get("DATA_DIR", "data")
MODEL_DIR = os.environ.get("MODEL_DIR", "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app = FastAPI(title="Florence-2 Backend")

class TrainParams(BaseModel):
    epochs: int = 1
    batch_size: int = 2
    task_type: str = "DocVQA"

@app.post("/upload")
async def upload_dataset(image: UploadFile = File(...), annotation: UploadFile = File(...)):
    """Upload an image and its annotation."""
    image_path = os.path.join(DATA_DIR, image.filename)
    ann_path = os.path.join(DATA_DIR, os.path.splitext(image.filename)[0] + ".json")
    with open(image_path, "wb") as img_f:
        shutil.copyfileobj(image.file, img_f)
    with open(ann_path, "wb") as ann_f:
        shutil.copyfileobj(annotation.file, ann_f)
    return {"status": "ok"}

@app.post("/train")
async def train(params: TrainParams):
    """Trigger training using scripts/train.py."""
    cmd = ["python", "scripts/train.py", "--dataset_folder", DATA_DIR,
           "--epochs", str(params.epochs), "--batch_size", str(params.batch_size),
           "--task_type", params.task_type]
    proc = subprocess.Popen(cmd)
    return {"pid": proc.pid}

@app.get("/status")
async def status(pid: int):
    """Check if a process with pid is running."""
    alive = os.path.exists(f"/proc/{pid}")
    return {"running": alive}

class EvalRequest(BaseModel):
    task_prompt: str
    text_input: str
    image_name: str
    model_dir: Optional[str] = MODEL_DIR

@app.post("/evaluate")
async def evaluate(req: EvalRequest):
    image_path = os.path.join(DATA_DIR, req.image_name)
    cmd = ["python", "scripts/val.py", "--task_prompt", req.task_prompt,
           "--text_input", req.text_input, "--image_path", image_path,
           "--model_dir", req.model_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return JSONResponse(status_code=500, content={"error": proc.stderr})
    return {"output": proc.stdout}
