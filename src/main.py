from prediction import predict_amiloidosis, preprocess_amiloidosis, preprocess_sclerosis, read_imagefile, predict_sclerosis
from fastapi import FastAPI
from fastapi import UploadFile, File
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
import io
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
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
    amiloidosis_array = preprocess_amiloidosis(image)
    amiloidosis_prediction = predict_amiloidosis(amiloidosis_array)
    sclerosis_array = preprocess_sclerosis(image)
    sclerosis_prediction = predict_sclerosis(sclerosis_array)
    if amiloidosis_prediction[0][0] > 0.5:
        retorno_a = "Amiloidose."
    else:
        retorno_a = "Sem amiloidose."
    if sclerosis_prediction[0][0] > 0.5:
        retorno_s = "Esclerose."
    else:
        retorno_s = "Sem esclerose."
    retorno = retorno_a + retorno_s
    return retorno


@app.post("/images/")
async def create_upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename}


@app.post("/return_image")
def image_endpoint():
    return FileResponse("public\\assets\\favicon.png")


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='localhost')
