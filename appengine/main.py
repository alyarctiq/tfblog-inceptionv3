#!/usr/local/bin/python
from flask import Flask
from flask import render_template
from flask import request, redirect
import googleapiclient.discovery
import json
import numpy as np
from keras import applications
from keras.preprocessing import image
from PIL import Image
import urllib.request
from keras.applications.inception_v3 import decode_predictions
from keras.applications.inception_v3 import preprocess_input
from io import BytesIO
import urllib


# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)

def predict_json(project, region, model, instances, version=None):
    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])
    return response['predictions']



def loadImage(URL):
    with urllib.request.urlopen(URL) as url:
        img = image.load_img(BytesIO(url.read()), target_size=(150,150))
    return img

def load_image(filename):
    with open(filename,'rb') as f:
        return np.array(f.read())

def decode(predictions):
     pred_arr = np.expand_dims(np.array(predictions), axis=0)
     decoded = decode_predictions(pred_arr, top=5)[0]
     # convert numpy dtypes to python native types
     #return [(t[0], t[1], t[2].item()) for t in decoded]
     return decoded

@app.route('/predict', methods=['POST'])
def form():
    labels=[]
    data=[]
    url=request.form['url']
    ### Change These To Match You Project ###
    project="tf-blog"
    region="uscentral1"
    model="tfblog"
    version="v3"
    #img = loadImage("https://upload.wikimedia.org/wikipedia/commons/d/dc/Fromia_monilis_%28Seastar%29.jpg")
    img = loadImage(url)
    w, h = img.size
    print(w,h)
    s = min(w, h)
    y = (h - s) // 2
    x = (w - s) // 2
    img = img.crop((x, y, s, s))
    #imshow(np.asarray(img))
    #img.save("test.jpg",optimize=True,quality=5)
    #img.save("test.jpg")

    np_img = image.img_to_array(img)
    img_batch = np.expand_dims(np_img, axis=0)
    pre_processed = preprocess_input(img_batch)
    print(pre_processed.shape)
    pre_processed=pre_processed.reshape(150,150,3)
    payload = [pre_processed.tolist()]
    features=predict_json(project, region, model, payload , version)
    for result in features:
         for k, v in result.items():
            preds=np.array(v) 
    #data=data.reshape(1,1000)
    #results=decode_predictions(data, top=10)
    results=decode(preds)
    for t in results:
        labels.append(t[1])
        data.append('%.08f' % (t[2]*100))
    return render_template('index.html', data=data, labels=labels)

#form()
@app.route('/')
def hello():
     return render_template('index.html')

if __name__ == '__main__':
     # This is used when running locally only. When deploying to Google App
     # Engine, a webserver process such as Gunicorn will serve the app. This
     # can be configured by adding an `entrypoint` to app.yaml.
     app.run(host='127.0.0.1', port=8080, debug=True)
