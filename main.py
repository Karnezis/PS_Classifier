from prediction import predict, preprocess, read_imagefile
from fastapi import FastAPI
from fastapi import UploadFile, File
import uvicorn

app = FastAPI()


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

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='localhost')
