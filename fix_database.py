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
        # Check if class_id column exists
        cursor.execute("DESCRIBE detections;")
        columns = cursor.fetchall()
        column_names = [col[0] for col in columns]
        
        if 'class_id' not in column_names:
            print("Adding class_id column...")
            cursor.execute("ALTER TABLE detections ADD COLUMN class_id FLOAT DEFAULT 0 AFTER bbox;")
            connection.commit()
            print("✓ class_id column added successfully!")
        else:
            print("✓ class_id column already exists")
        
        # Show final table structure
        print("\nFinal table structure:")
        cursor.execute("DESCRIBE detections;")
        for row in cursor.fetchall():
            print(f"  {row[0]}: {row[1]}")
            
finally:
    connection.close()

print("\nDatabase fix complete! Restart your backend now.")

