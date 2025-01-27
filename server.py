from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import subprocess
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

@app.get("/")
async def root():
    
    return RedirectResponse("/")

@app.on_event("startup")
async def start_streamlit():
    try:
        subprocess.Popen(
            ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
        )
        logging.info("Streamlit app started successfully.")
    except Exception as e:
        logging.error(f"Error running Streamlit: {e}")
