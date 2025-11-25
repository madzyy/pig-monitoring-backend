import uuid
import datetime
from config import USE_DYNAMODB, MYSQL_URL, AWS_REGION, DYNAMODB_TABLE

# -----------------------------
# MySQL (SQLAlchemy)
# -----------------------------
if not USE_DYNAMODB:
    from sqlalchemy import create_engine, Column, String, Float, DateTime, JSON
    from sqlalchemy.orm import declarative_base, sessionmaker

    engine = create_engine(MYSQL_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    class Detection(Base):
        __tablename__ = "detections"

        id = Column(String(255), primary_key=True)
        filename = Column(String(255))
        confidence = Column(Float)
        bbox = Column(JSON)  # [x, y, w, h]
        class_id = Column(Float, default=0)  # Behavior class: 0=lying, 1=sleeping, 2=investigating, 3=eating, 4=walking, 5=mounted
        timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    def create_tables():
        Base.metadata.create_all(bind=engine)
    
    # Define stub DynamoDB functions for MySQL mode
    def save_dynamodb_item(item: dict):
        raise NotImplementedError("DynamoDB is disabled. Use MySQL.")
    
    def get_all_dynamodb_items():
        raise NotImplementedError("DynamoDB is disabled. Use MySQL.")
    
    def get_dynamodb_item(det_id: str):
        raise NotImplementedError("DynamoDB is disabled. Use MySQL.")
    
    def delete_dynamodb_item(det_id: str):
        raise NotImplementedError("DynamoDB is disabled. Use MySQL.")


# -----------------------------
# DynamoDB
# -----------------------------
else:
    import boto3

    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    table = dynamodb.Table(DYNAMODB_TABLE)

    def save_dynamodb_item(item: dict):
        table.put_item(Item=item)

    def get_all_dynamodb_items():
        return table.scan().get("Items", [])

    def get_dynamodb_item(det_id: str):
        return table.get_item(Key={"id": det_id}).get("Item")

    def delete_dynamodb_item(det_id: str):
        table.delete_item(Key={"id": det_id})
    
    # Define stub MySQL objects for DynamoDB mode
    SessionLocal = None
    Detection = None
    def create_tables():
        pass  # No-op in DynamoDB mode
