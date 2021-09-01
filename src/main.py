from prediction import predict, preprocess, read_imagefile
from fastapi import FastAPI
from fastapi import UploadFile, File
from starlette.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
CORSMiddleware,
allow_origins=["*"], # Allows all origins
allow_credentials=True,
allow_methods=["*"], # Allows all methods
allow_headers=["*"], # Allows all headers
)

@app.get('/')
async def hello_world():
    return "Hello world!"

@app.get('/index')
async def hello_world(name: str):
    return f"Hello {name}!"


@app.post('/api/predict')
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    pre_processed_array = preprocess(image)
    prediction = predict(pre_processed_array)
    if prediction[0][0] > 0.5:
        return "Amiloidose."
    else:
        return "Saudável."

@app.post("/images/")
async def create_upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename}

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='localhost')
