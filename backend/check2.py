from app.db.session import SessionLocal 
from app.models.fileModels import File 
db = SessionLocal() 
files = db.query(File).filter(File.post_idx == '147e5fca-0a8a-4535-b7f2-d8bfb2d7d049').all() 
print(len(files)) 
[print(f.file_type, f.file_path) for f in files] 
db.close() 
