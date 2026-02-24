from fastapi import APIRouter, UploadFile, File
from app.services.grip.gripService import GripService

router = APIRouter(prefix="/api/grip", tags=["Grip Analysis"])

@router.post("/analyze")
async def analyze_grip(file: UploadFile = File(...)):
    contents = await file.read()
    result = GripService.predict_grip(contents)
    return result