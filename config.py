import os

# Switch to DynamoDB when deploying to AWS
USE_DYNAMODB = os.getenv("USE_DYNAMODB", "false").lower() == "true"

# MySQL settings (local dev) - XAMPP default has no password for root
MYSQL_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:@localhost/livestock")

# DynamoDB settings (AWS)
AWS_REGION = "us-east-1"
DYNAMODB_TABLE = "LivestockDetections"
