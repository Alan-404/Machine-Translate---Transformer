from fastapi import FastAPI
from predict import Predictor
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:4200"
]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


predictor = Predictor()

class InputTranslate(BaseModel):
    input: str

@app.post('/translate')
def translate(data:InputTranslate):
    return {"result": predictor.predict(data.input)}