import pymysql

# Connect to MySQL
connection = pymysql.connect(
    host='localhost',
    user='root',
    password='',
    database='livestock'
)

try:
    with connection.cursor() as cursor:
        # Drop the old table
        print("Dropping old table...")
        cursor.execute("DROP TABLE IF EXISTS detections;")
        
        # Create new table with class_id column
        print("Creating new table with class_id column...")
        cursor.execute("""
            CREATE TABLE detections (
                id VARCHAR(255) PRIMARY KEY,
                filename VARCHAR(255),
                confidence FLOAT,
                bbox JSON,
                class_id FLOAT DEFAULT 0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        connection.commit()
        print("✓ Table recreated successfully!")
        
        # Show final table structure
        print("\nNew table structure:")
        cursor.execute("DESCRIBE detections;")
        for row in cursor.fetchall():
            print(f"  {row[0]}: {row[1]}")
            
except Exception as e:
    print(f"Error: {e}")
finally:
    connection.close()

print("\n✓ Database is ready! Restart your backend now.")

