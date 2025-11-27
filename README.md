# 🐖 Cloud AI Livestock Security & Monitoring System — Backend  
FastAPI • YOLO • ONNX Runtime • MySQL • AWS RDS • S3 • Docker

This backend powers an AI-driven livestock monitoring and security system designed to detect early signs of sickness, unusual animal behavior, and possible theft using computer vision and collar-based telemetry.

It provides:
- Real-time behavior classification using a YOLO model  
- REST APIs for livestock records, analytics, GPS tracking, and alerts  
- ONNX-based inference for optimized speed  
- Integration with AWS services (RDS, Aurora, S3)  
- Dockerized deployment

---

## 🚀 Features

### 🔍 AI & Computer Vision
- YOLO model trained on 25 epochs using a Mendeley pig dataset  
- Exported to **ONNX** for fast inference  
- Behavior detection: *standing, lying, walking, sleeping*  
- Designed to support health anomaly scoring

### 🧠 Backend Architecture
- FastAPI for API endpoints  
- ONNX Runtime for AI inference  
- MySQL / AWS RDS / Aurora for database storage  
- S3 bucket for image storage  
- CORS-enabled for React frontend

### 🗺 Livestock GPS & Security
- GPS tracking endpoints  
- Geofencing: detect out-of-bounds animals  
- Theft alerts & notifications  
- Temperature + health status from planned collar device

---

## 🧩 Project Structure

src/
├── api/
│ ├── inference.py
│ ├── animals.py
│ ├── gps.py
│ ├── analytics.py
│ ├── alerts.py
│ └── upload.py
├── models/
│ └── model.onnx
├── database/
│ └── connection.py
├── utils/
│ ├── preprocessing.py
│ └── geofence.py
└── main.py



---

## 🛠 Tech Stack

- **FastAPI**
- **Python 3**
- **ONNX Runtime**
- **MySQL / AWS RDS / Aurora**
- **Docker**
- **AWS S3**
- **Uvicorn**

---

## ⚙ Installation & Setup

### Clone the repository
```sh
git clone https://github.com/madzyy/pig-monitoring-backend.git
cd pig-monitoring-backend
```

##Install dependencies
```sh
pip install -r requirements.txt
```
##Set environment variables

##Create .env:
```sh
DB_HOST=
DB_USER=
DB_PASSWORD=
DB_NAME=
AWS_S3_BUCKET=
AWS_REGION=
```

##Run locally
```sh
uvicorn main:app --reload
```

##🐳 Docker Deployment
##Build image
```sh
docker build -t pig-backend .
```

##Run container
```sh
docker run -p 8000:8000 --env-file .env pig-backend
```

##☁ AWS Deployment (Summary)

- Backend deployed with Docker

- ONNX model stored locally in container

- Database hosted in AWS RDS / Aurora

- Images uploaded to AWS S3
