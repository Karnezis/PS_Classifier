from starlette.requests import Request
from prediction import predict_amiloidosis, preprocess_amiloidosis, preprocess_sclerosis, read_imagefile, predict_sclerosis, preprocess_hiper, predict_hiper
from gradcam.visualizer import Visualizer
from fastapi import Depends, FastAPI
from fastapi import UploadFile, File
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
import io
import uuid
import requests
import json
import uvicorn
import time
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
    start_time = time.time()
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    gradcam = Visualizer()
    myuuid = uuid.uuid4()
    file_location = f"src/images/{myuuid}{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    image = Image.open(file_location, 'r')
    '''amiloidosis_array = preprocess_amiloidosis(image)
    amiloidosis_prediction = predict_amiloidosis(amiloidosis_array)'''
    sclerosis_array = preprocess_sclerosis(image)
    sclerosis_prediction = predict_sclerosis(sclerosis_array)
    hiper_array = preprocess_hiper(image)
    hiper_prediction = predict_hiper(hiper_array)
    views = []
    '''if amiloidosis_prediction[0][0] > 0.5:
        retorno_a = "Amiloidose."
        fp_view = gradcam.visualize(
            file_location, 1, 'middle', -1, 'CAM_IMAGE_JET', False)
        views.append(fp_view)
    else:
        retorno_a = "Sem amiloidose."'''
    if sclerosis_prediction[0][0] > 0.5:
        retorno_s = "Esclerose."
        fp_view = gradcam.visualize(
            file_location, 2, 'middle', -1, 'CAM_IMAGE_JET', False)
        views.append(fp_view)
    else:
        retorno_s = "Sem esclerose."
    if hiper_prediction == 1:
        fp_view = gradcam.visualize(
            file_location, 0, 'middle', -1, 'CAM_IMAGE_JET', False)
        views.append(fp_view)
        retorno_h = 'Endocapilar e mesangial.'
    elif hiper_prediction == 2:
        fp_view = gradcam.visualize(
            file_location, 0, 'middle', -1, 'CAM_IMAGE_JET', False)
        views.append(fp_view)
        retorno_h = 'Mesangial.'
    elif hiper_prediction == 3:
        fp_view = gradcam.visualize(
            file_location, 0, 'middle', -1, 'CAM_IMAGE_JET', False)
        views.append(fp_view)
        retorno_h = 'Endocapilar'
    else:
        retorno_h = 'No'
    myuuid = uuid.uuid4()
    '''if(len(views) > 1):
        im1 = Image.open(views[0])
        im2 = Image.open(views[1])
        view_location = f"src/images/{myuuid}-view.png"
        Image.blend(im1, im2, 0.5).save(view_location, bitmap_format='png')
        retorno_v = view_location
        with open(retorno_v, 'rb') as fh:
            buf = io.BytesIO(fh.read())
    else:'''
    '''retorno_v = views[0]
    retorno = {"Esclerose": retorno_s, "Hipercelularidade": retorno_h,
               "File": file.filename, "Time": "--- %s seconds ---" % (time.time() - start_time),
               "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Credentials": "true"}
    print("--- %s seconds ---" % (time.time() - start_time))
    return Response(content=retorno_v.getvalue(), headers=retorno, media_type="image/png")'''
    return {"Esclerose": retorno_s, "Hipercelularidade": retorno_h,
            "File": file.filename, "Time": "--- %s seconds ---" % (time.time() - start_time),
            "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Credentials": "true"}


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
    uvicorn.run(app, port=8389, host='localhost')
