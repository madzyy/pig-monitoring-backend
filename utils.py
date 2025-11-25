# utils.py
import cv2
import numpy as np

def letterbox(img, new_size=640, color=(114,114,114)):
    """Resize image to square new_size preserving aspect ratio, return padded image, scale, and pad offsets."""
    h, w = img.shape[:2]
    r = new_size / max(h, w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    dw = new_size - new_unpad[0]
    dh = new_size - new_unpad[1]
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img_padded, r, (left, top)

def preprocess_image_bytes(image_bytes, input_size=640):
    """Return original image (BGR), model_input (1,3,H,W float32), and metadata for mapping boxes."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError("Could not decode image")
    img_padded, scale, pad = letterbox(img, new_size=input_size)
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    tensor = np.transpose(img_norm, (2,0,1))[None, ...]  # 1,3,H,W
    return img, tensor.astype(np.float32), {"scale": scale, "pad": pad, "orig_shape": img.shape[:2]}

def xyxy_to_orig(box, meta):
    """Map box from padded/resized space back to original image coords.
       box = [x1,y1,x2,y2] using same scale/pad as letterbox output"""
    left, top = meta['pad']
    scale = meta['scale']
    x1 = (box[0] - left) / scale
    y1 = (box[1] - top) / scale
    x2 = (box[2] - left) / scale
    y2 = (box[3] - top) / scale
    return [int(x1), int(y1), int(x2), int(y2)]

def non_max_suppression(boxes, scores, iou_thresh=0.45):
    """Boxes Nx4 (x1,y1,x2,y2), scores N -> indices to keep"""
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:,0]; y1 = boxes[:,1]; x2 = boxes[:,2]; y2 = boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

def inspect_onnx_session(sess):
    """Print info about input/output tensors of onnx runtime session."""
    print("=== ONNX model inputs ===")
    for i in sess.get_inputs():
        print(i.name, i.shape, i.type)
    print("=== ONNX model outputs ===")
    for o in sess.get_outputs():
        print(o.name, o.shape, o.type)
