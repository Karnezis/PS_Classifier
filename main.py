from prediction import predict, preprocess, read_imagefile
from fastapi import FastAPI
from fastapi import UploadFile, File
import uvicorn

app = FastAPI()


@app.get('/index')
async def hello_world(name: str):
    return f"Hello {name}!"


@app.post('/api/predict')
async def predict_image(file: UploadFile = File(...)):
    image = read_imagefile(await file)
    pre_processed_array = preprocess(image)
    prediction = predict(pre_processed_array)
    print(prediction)
    return prediction

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='localhost')
