USE livestock;
ALTER TABLE detections ADD COLUMN class_id FLOAT DEFAULT 0;
DESCRIBE detections;

