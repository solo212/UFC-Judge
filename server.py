from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import subprocess

app = FastAPI()

@app.get("/")
async def root():
    
    return RedirectResponse("/app")



@app.on_event("startup")
async def start_streamlit():
    try:
       
        subprocess.Popen(
            ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,  
        )
    except Exception as e:
        print(f"Error running Streamlit: {e}")

