from fastapi import FastAPI
from fastapi.responses import JSONResponse
import datetime


app = FastAPI()


@app.get("/time")
def time():
    current_time = datetime.datetime.utcnow()
    iso_format = current_time.strftime("%Y-%m-%T")

    return {"timestamp": iso_format}
