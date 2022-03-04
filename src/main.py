from prediction import predict_amiloidosis, preprocess_amiloidosis, preprocess_sclerosis, read_imagefile, predict_sclerosis
from gradcam.visualizer import Visualizer
from fastapi import FastAPI
from fastapi import UploadFile, File
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
import io
import uuid
import requests
import json
import uvicorn
from PIL import Image
from starlette.responses import Response

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
    gradcam = Visualizer()
    myuuid = uuid.uuid4()
    file_location = f"src/images/{myuuid}{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    image = Image.open(file_location, 'r')
    amiloidosis_array = preprocess_amiloidosis(image)
    amiloidosis_prediction = predict_amiloidosis(amiloidosis_array)
    sclerosis_array = preprocess_sclerosis(image)
    sclerosis_prediction = predict_sclerosis(sclerosis_array)
    views = []
    if amiloidosis_prediction[0][0] > 0.5:
        retorno_a = "Amiloidose."
        '''fp_view = gradcam.visualize(
            file_location, 1, 'middle', -1, 'CAM_IMAGE_JET', False)
        views.append(fp_view)'''
    else:
        retorno_a = "Sem amiloidose."
    if sclerosis_prediction[0][0] > 0.5:
        retorno_s = "Esclerose."
        fp_view = gradcam.visualize(
            file_location, 2, 'middle', -1, 'CAM_IMAGE_JET', False)
        views.append(fp_view)
    else:
        retorno_s = "Sem esclerose."
    myuuid = uuid.uuid4()
    '''if(len(views) > 1):
        im1 = Image.open(views[0])
        im2 = Image.open(views[1])
        view_location = f"src/images/{myuuid}{file.filename}-view"
        Image.blend(im1, im2, 0.5).save(view_location)
        retorno_v = view_location
    else:
        retorno_v = views[0]'''
    retorno = {"Amiloidose": retorno_a, "Esclerose": retorno_s}
    return Response(content=fp_view.getvalue(), headers=retorno, media_type="image/png")


@app.post("/images/")
async def create_upload_file(file: UploadFile = File(...)):
    gradcam = Visualizer()
    myuuid = uuid.uuid4()
    file_location = f"src/images/{myuuid}{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    '''gradcam.visualize(file_location, model, 'middle',
                      self.visLabel, 'CAM_IMAGE_JET', self.visGuided)'''
    return {"info": f"file '{file.filename}' saved at '{file_location}'"}


@app.post("/return_image")
def image_endpoint():
    return {"file": "FileResponse(\"public\\\\assets\\\\favicon.png\")"}


@app.post("/upload-file/")
async def create_upload_file(uploaded_file: UploadFile = File(...)):
    file_location = f"src/images/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())
    return {"info": f"file '{uploaded_file.filename}' saved at '{file_location}'"}


@app.post("/return-images/")
def get_text(image_list, url):
    try:
        image_data = []
        for image in image_list:
            # ('files',(image_name,open image,type))
            image_data.append(
                ('files', (image.split('/')[-1], open(image, 'rb'), 'image/png')))
        response = requests.post(url, files=image_data)
        return json.loads(response.text)
    except Exception as er:
        print("error occured")
        return "{} error occured".format(er)


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='localhost')
