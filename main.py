#!/usr/local/bin/python
#from flask import Flask
#from flask import render_template
#from flask import request, redirect
import googleapiclient.discovery
import json
import numpy as np
from keras import applications
from keras.preprocessing import image
from PIL import Image
import urllib.request

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
#app = Flask(__name__)

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
    print response['predictions']
    return response['predictions']

from io import BytesIO
import urllib

def loadImage(URL):
    with urllib.request.urlopen(URL) as url:
        img = image.load_img(BytesIO(url.read()), target_size=(299, 299))
    return img

def load_image(filename):
    with open(filename) as f:
        return np.array(f.read())

# app.route('/predict', methods=['POST'])
def form():
    ### Change These To Match You Project ###
    project="tf-blog"
    region="uscentral1"
    model="tfblog"
    img = loadImage("https://upload.wikimedia.org/wikipedia/commons/d/dc/Fromia_monilis_%28Seastar%29.jpg")
    img.save("test.jpg")
    version="v3"
    predict_json(project, region, model, [{load_image("test.jpg").tolist()}], version)


form()
# @app.route('/')
# def hello():
#     return render_template('index.html')

# if __name__ == '__main__':
#     # This is used when running locally only. When deploying to Google App
#     # Engine, a webserver process such as Gunicorn will serve the app. This
#     # can be configured by adding an `entrypoint` to app.yaml.
#     app.run(host='127.0.0.1', port=8080, debug=True)