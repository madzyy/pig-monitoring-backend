import uuid
import numpy as np
import cv2
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import datetime

from config import USE_DYNAMODB
from database import (
    SessionLocal, Detection, create_tables,
    save_dynamodb_item, get_all_dynamodb_items,
    get_dynamodb_item, delete_dynamodb_item
)
from models import DetectionResponse, PredictionResponse

app = FastAPI(title="Livestock YOLO API")

# Add CORS middleware to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Initialize MySQL tables
# -------------------------
if not USE_DYNAMODB:
    create_tables()

# -------------------------
# Load YOLO ONNX model
# -------------------------
print("Loading ONNX model from: models/best.onnx")
session = ort.InferenceSession("models/best.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # [1,3,640,640]
output_names = [o.name for o in session.get_outputs()]
output_shapes = [o.shape for o in session.get_outputs()]
print(f"Model loaded successfully!")
print(f"Input: {input_name} {input_shape}")
print(f"Outputs: {list(zip(output_names, output_shapes))}")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "models/best.onnx",
        "input": {"name": input_name, "shape": input_shape},
        "outputs": [{"name": n, "shape": s} for n, s in zip(output_names, output_shapes)]
    }

# Helper: preprocess image
def preprocess(img):
    print(f"Original image shape: {img.shape}")
    img_resized = cv2.resize(img, (640, 640))
    print(f"Resized image shape: {img_resized.shape}")
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    print(f"Normalized image range: {img_norm.min():.3f} to {img_norm.max():.3f}")
    img_trans = np.transpose(img_norm, (2, 0, 1))
    img_batch = img_trans[np.newaxis, :]
    print(f"Final input shape: {img_batch.shape}")
    return img_batch

# Helper: save a detection
def save_detection(filename: str, confidence: float, bbox: list, class_id: int = 0):
    detection_id = str(uuid.uuid4())

    if USE_DYNAMODB:
        item = {
            "id": detection_id,
            "filename": filename,
            "confidence": float(confidence),
            "bbox": list(map(float, bbox)),
            "class_id": int(class_id),
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        save_dynamodb_item(item)
        return item

    else:
        db = SessionLocal()
        det = Detection(
            id=detection_id,
            filename=filename,
            confidence=float(confidence),
            bbox=bbox,
            class_id=int(class_id),
        )
        db.add(det)
        db.commit()
        db.refresh(det)
        
        # Debug: check what's being saved
        print(f"Saved detection to DB - ID: {det.id}, class_id: {det.class_id}, bbox: {det.bbox}")
        
        db.close()
        return det


# -------------------------
# Prediction Endpoint
# -------------------------
@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read and decode image
        img_bytes = await file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")

        # Preprocess and run inference
        inp = preprocess(img)
        outputs = session.run(None, {input_name: inp})
        
        # Debug: print output shapes
        print(f"Number of outputs: {len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"Output {i} shape: {out.shape}")
        
        # Parse ONNX outputs - handle different YOLOv8 output formats
        detections = []
        raw_output = outputs[0]
        
        print(f"Raw output shape: {raw_output.shape}")
        print(f"Raw output min/max: {raw_output.min()}/{raw_output.max()}")
        
        # YOLOv8 typically outputs shape [1, features, num_boxes] 
        # Example: [1, 10, 8400] means 8400 predictions with 10 features each (4 box + 6 classes)
        # We need to transpose to [1, num_boxes, features]
        if raw_output.ndim == 3:
            # Shape: [batch, features, boxes] -> transpose to [batch, boxes, features]
            # If shape is [1, 10, 8400], features (10) < boxes (8400), so we need to transpose
            if raw_output.shape[1] < raw_output.shape[2]:
                # [1, 10, 8400] -> [1, 8400, 10]
                raw_output = raw_output.transpose(0, 2, 1)
                print(f"Transposed from shape with features axis 1 to axis 2")
            
            # Now shape should be [1, num_boxes, features]
            predictions = raw_output[0]  # Remove batch dimension
            print(f"Predictions shape after processing: {predictions.shape}")
            print(f"Number of predictions: {predictions.shape[0]}")
            print(f"Number of features per prediction: {predictions.shape[1]}")
            
            # YOLOv8 format: [x_center, y_center, width, height, class_scores...]
            detection_count = 0
            for idx, pred in enumerate(predictions):
                # Extract box coordinates (first 4 values)
                x_center, y_center, w, h = pred[:4]
                
                # Extract class scores (remaining values) - should be 6 classes for pig behaviors
                class_scores = pred[4:]
                
                max_score = float(np.max(class_scores))
                class_id = int(np.argmax(class_scores))
                
                # Filter by confidence (lowered threshold for pig behaviors)
                if max_score < 0.25:  # Lowered from 0.4 to 0.25 for better detection
                    continue
                
                detection_count += 1
                
                # Show details for ALL accepted detections
                behaviors = ['Lying', 'Sleeping', 'Investigating', 'Eating', 'Walking', 'Mounted']
                print(f"\n✓ Detection #{detection_count}:")
                print(f"  Confidence: {max_score:.4f}")
                print(f"  Class breakdown:")
                for i, (behavior, score) in enumerate(zip(behaviors, class_scores)):
                    marker = " ← PREDICTED" if i == class_id else ""
                    print(f"    {i}. {behavior:15s}: {score:.4f}{marker}")
                
                # Check if coordinates are normalized (0-1) and convert to pixels if needed
                # Most YOLOv8 models output pixel coordinates directly when trained with 640x640
                if x_center <= 1.0 and y_center <= 1.0 and w <= 1.0 and h <= 1.0:
                    # Normalized coordinates - scale to 640x640
                    print(f"  Detected normalized coordinates, scaling to 640x640")
                    x_center *= 640
                    y_center *= 640
                    w *= 640
                    h *= 640
                
                # Convert from center format to corner format
                x1 = float(x_center - w / 2)
                y1 = float(y_center - h / 2)
                bbox = [x1, y1, float(w), float(h)]
                
                
                # Try to save to database, fall back to dict if it fails
                try:
                    saved = save_detection(file.filename, max_score, bbox, class_id)
                    detections.append(saved)
                except Exception as db_error:
                    print(f"  ⚠️  Database save failed: {db_error}")
                    print(f"  → Returning detection without saving to DB")
                    # Create detection dict without saving
                    detection_dict = {
                        "id": str(uuid.uuid4()),
                        "filename": file.filename,
                        "confidence": float(max_score),
                        "bbox": bbox,
                        "class_id": int(class_id),
                        "timestamp": datetime.datetime.utcnow().isoformat()
                    }
                    detections.append(detection_dict)
            
            print(f"\nTotal detections accepted: {len(detections)}")
            if len(detections) == 0:
                print("⚠️  No detections passed the confidence threshold (0.25)")
                print("   This could mean:")
                print("   1. No pigs in the image")
                print("   2. Pigs are too small or unclear")
                print("   3. Model confidence is low - try a clearer image")
        
        else:
            # Fallback for other formats
            print(f"Unexpected output shape: {raw_output.shape}")
            raise ValueError(f"Unsupported model output shape: {raw_output.shape}")

        # Debug: Print what we're returning
        print(f"\n=== RETURNING RESPONSE ===")
        print(f"Filename: {file.filename}")
        print(f"Number of detections: {len(detections)}")
        for i, det in enumerate(detections):
            print(f"Detection {i}: {det}")
        
        return {
            "filename": file.filename,
            "detections": detections
        }
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


# -------------------------
# Admin: list detections
# -------------------------
@app.get("/detections", response_model=List[DetectionResponse])
def list_detections():
    if USE_DYNAMODB:
        return get_all_dynamodb_items()

    db = SessionLocal()
    items = db.query(Detection).all()
    db.close()
    return items


# -------------------------
# Admin: single detection
# -------------------------
@app.get("/detections/{det_id}", response_model=DetectionResponse)
def get_detection(det_id: str):
    if USE_DYNAMODB:
        return get_dynamodb_item(det_id)

    db = SessionLocal()
    item = db.query(Detection).filter(Detection.id == det_id).first()
    db.close()
    return item


# -------------------------
# Admin: delete detection
# -------------------------
@app.delete("/detections/{det_id}")
def delete_detection_api(det_id: str):
    if USE_DYNAMODB:
        delete_dynamodb_item(det_id)
        return {"deleted": det_id}

    db = SessionLocal()
    db.query(Detection).filter(Detection.id == det_id).delete()
    db.commit()
    db.close()
    return {"deleted": det_id}
