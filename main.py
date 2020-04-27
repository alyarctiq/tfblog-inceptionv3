
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
    return response['predictions']

from io import BytesIO
import urllib

def loadImage(URL):
    with urllib.request.urlopen(URL) as url:
        img = image.load_img(BytesIO(url.read()), target_size=(224,224))
    return img

def load_image(filename):
    with open(filename,'rb') as f:
        return np.array(f.read())

def decode(predictions):
     pred_arr = np.expand_dims(np.array(predictions), axis=0)
     decoded = decode_predictions(pred_arr, top=5)[0]
     # convert numpy dtypes to python native types
     return [(t[0], t[1], t[2].item()) for t in decoded]

# app.route('/predict', methods=['POST'])
def form():
    ### Change These To Match You Project ###
    project="tf-blog"
    region="uscentral1"
    model="tfblog"
    l0=[]
    img = loadImage("https://upload.wikimedia.org/wikipedia/commons/d/dc/Fromia_monilis_%28Seastar%29.jpg")
    #img.save("test.jpg",optimize=True,quality=5)
    img.save("test.jpg")
    version="v3"
    #instance=np.asarray(img)
    #imgA = image.img_to_array(image.load_img("test.jpg",target_size=(299, 299))) / 255.
    imgA = image.img_to_array(image.load_img("test.jpg"))
    #payload = {
    #        "instances": [{'input_image': imgA.tolist()}]
    #        }
    payload = [imgA.tolist()]
    features=predict_json(project, region, model, payload , version)
    #print(decode_predictions(np.array(features[0]))['prediction'][0])
    for result in features:
         for k, v in result.items():
            data=np.array(v)
    data1=data.reshape(1,1000)
    #print(data1.shape)
    print(decode_predictions(data1, top=5))

form()
# @app.route('/')
# def hello():
#     return render_template('index.html')

# if __name__ == '__main__':
#     # This is used when running locally only. When deploying to Google App
#     # Engine, a webserver process such as Gunicorn will serve the app. This
#     # can be configured by adding an `entrypoint` to app.yaml.
#     app.run(host='127.0.0.1', port=8080, debug=True)