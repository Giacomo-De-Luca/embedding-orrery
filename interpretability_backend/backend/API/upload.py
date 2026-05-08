import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter()

# Define uploads directory
# interpretability_backend/resources/uploads
UPLOAD_DIR = Path(__file__).parent.parent.parent / "resources" / "uploads"


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file to the server for processing.
    Returns the absolute path to the saved file.
    """
    try:
        # DB path might not exist yet if clean install, but resources should
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        # Sanitize filename (basic)
        filename = Path(file.filename).name
        target_path = UPLOAD_DIR / filename

        # Save file
        with target_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {"filePath": str(target_path.absolute())}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")
