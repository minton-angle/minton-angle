from app.db.session import SessionLocal 
from app.models.analysisModels import Analysis 
db = SessionLocal() 
a = db.query(Analysis).filter(Analysis.post_idx == '147e5fca-0a8a-4535-b7f2-d8bfb2d7d049').first() 
print('score_json:', a.score_json if a else 'None') 
db.close() 
