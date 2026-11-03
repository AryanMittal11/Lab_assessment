from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

from routes.extract import router as extract_router
from routes.upload import router as upload_router

load_dotenv(Path(__file__).resolve().parent / ".env")

app = FastAPI()

app.include_router(upload_router)
app.include_router(extract_router)


@app.get("/")
def read_root():
    return {"message": "Hello World"}