# main.py
import os
import io
import time
import uuid
import json
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import onnxruntime as ort
from utils import preprocess_image_bytes, xyxy_to_orig, non_max_suppression, inspect_onnx_session
import boto3

# CONFIG - set env vars or change here
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.onnx")
INPUT_SIZE = int(os.environ.get("INPUT_SIZE", 640))
USE_AWS = os.environ.get("USE_AWS", "false").lower() == "true"
S3_BUCKET = os.environ.get("S3_BUCKET", "")
DYNAMO_TABLE = os.environ.get("DYNAMO_TABLE", "")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
LOCAL_UPLOAD_DIR = os.environ.get("LOCAL_UPLOAD_DIR", "uploads")

if USE_AWS:
    s3 = boto3.client("s3", region_name=AWS_REGION)
    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    table = dynamodb.Table(DYNAMO_TABLE)
else:
    os.makedirs(LOCAL_UPLOAD_DIR, exist_ok=True)

# Load ONNX model
print("Loading ONNX model from:", MODEL_PATH)
sess = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
inspect_onnx_session(sess)  # prints shapes for debugging

input_name = sess.get_inputs()[0].name
output_names = [o.name for o in sess.get_outputs()]
print("input_name:", input_name, "outputs:", output_names)

app = FastAPI(title="Livestock ONNX Inference API")

class InferResponse(BaseModel):
    event_id: str
    detections: list

def save_image_local(event_id, data):
    path = os.path.join(LOCAL_UPLOAD_DIR, f"{event_id}.jpg")
    with open(path, "wb") as f:
        f.write(data)
    return path

def upload_to_s3(key, data, content_type="image/jpeg"):
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=data, ContentType=content_type)
    url = s3.generate_presigned_url('get_object', Params={'Bucket': S3_BUCKET, 'Key': key}, ExpiresIn=604800)
    return url

def store_event_dynamo(item):
    table.put_item(Item=item)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

@app.post("/infer", response_model=InferResponse)
async def infer(
    file: UploadFile = File(...),
    device_id: Optional[str] = Form(None),
    gps_lat: Optional[float] = Form(None),
    gps_lon: Optional[float] = Form(None),
    temperature: Optional[float] = Form(None)
):
    # read bytes
    body = await file.read()
    # preprocess
    orig_img, input_tensor, meta = preprocess_image_bytes(body, input_size=INPUT_SIZE)
    # run ONNX
    outputs = sess.run(None, {input_name: input_tensor})
    # ---------- IMPORTANT ----------
    # Inspect outputs shapes printed earlier to adapt this decode block.
    # We handle two common cases:
    # 1) Model outputs already in detection format: [N,6] rows = [x1,y1,x2,y2,score,class]
    # 2) Model outputs as [1, boxes, 6] -> squeeze to [boxes,6]
    onnx_out = None
    # find first output that looks like detection array
    for o in outputs:
        arr = o
        if isinstance(arr, list):
            continue
        if arr.ndim == 2 and arr.shape[1] >= 6:
            onnx_out = arr
            break
        if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] >= 6:
            onnx_out = arr[0]
            break
    if onnx_out is None:
        # fallback: take first output
        onnx_out = outputs[0].squeeze()
    # Filter detections by confidence threshold
    conf_thres = 0.25
    detections_all = []
    for row in onnx_out:
        # try to robustly parse common layouts
        # row could be [x1,y1,x2,y2,conf,cls] or [x_center,y_center,w,h,conf,cls]
        if len(row) >= 6:
            x1, y1, x2, y2, conf, cls = row[:6]
            # sanity: if values look like center/wh (0..1) try convert
            if 0 <= x1 <= 1 and 0 <= x2 <= 1 and (x2 <= 1):
                # probably normalized center format? attempt guess: center x,y,w,h in 0..1
                # but safer to skip conversion unless boxes small values -> skip for now
                pass
            if conf >= conf_thres:
                detections_all.append({"box":[float(x1), float(y1), float(x2), float(y2)],
                                       "score": float(conf),
                                       "class": int(cls)})
    # apply NMS in padded coordinate space
    boxes = [d["box"] for d in detections_all]
    scores = [d["score"] for d in detections_all]
    keep = non_max_suppression(boxes, scores, iou_thresh=0.45)
    detections = []
    for i in keep:
        det = detections_all[i]
        # convert to original image coords
        orig_box = xyxy_to_orig(det["box"], meta)
        detections.append({
            "bbox": orig_box,
            "score": det["score"],
            "class": det["class"]
        })
    # Save image + event
    event_id = str(uuid.uuid4())
    if USE_AWS:
        key = f"images/{event_id}.jpg"
        img_url = upload_to_s3(key, body)
    else:
        local_path = save_image_local(event_id, body)
        img_url = f"file://{os.path.abspath(local_path)}"
    event = {
        "event_id": event_id,
        "device_id": device_id or "unknown",
        "timestamp": int(time.time()),
        "gps": {"lat": gps_lat, "lon": gps_lon} if gps_lat and gps_lon else {},
        "temperature": temperature,
        "image_url": img_url,
        "detections": detections,
        "alert_sent": False
    }
    if USE_AWS:
        try:
            store_event_dynamo(event)
        except Exception as e:
            print("DynamoDB store error:", e)
    else:
        # write JSON log locally
        log_path = os.path.join(LOCAL_UPLOAD_DIR, f"{event_id}.json")
        with open(log_path, "w") as f:
            json.dump(event, f, indent=2)
    return {"event_id": event_id, "detections": detections}
