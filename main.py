from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Consider specific origins in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Added GET here
    allow_headers=["*"],
)

# Load model at startup
try:
    model = YOLO("yolov8n.pt")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise RuntimeError("Failed to initialize model")

# Simple GET endpoint for health check
@app.get("/")
async def root():
    return {"message": "Hello, FastAPI in settings!"}

# Object detection endpoint
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read and validate image
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
            
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Run inference
        results = model(image, conf=0.5)
        
        # Process results
        detections = []
        for box in results[0].boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": [float(x) for x in box.xyxy[0].tolist()]  # [x1, y1, x2, y2]
            })
            
        return {"detections": detections}
        
    except Exception as e:
        logger.error(f"Detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing image")