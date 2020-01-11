from flask import Flask, render_template, request
import cv2

import numpy as np
import re
import base64
from load import *

app = Flask(__name__)


global model, graph
model, graph = init()


@app.route('/')
def index():
	print("App Started")
	return render_template("index.html")


@app.route('/predict/', methods=['POST'])
def predict():
    print("Analysis Started")
    if request.method == "POST":
	    map_num_exp = { 0:"People", 1: "Object"}
	    imgData = request.get_data()
	    # print(imgData)
	    return ml(imgData)
    else:
    	return "Error!"    



def cropImage():
    im=Image.open("captured_image.png").convert('LA')
    # print(im.size) #320, 240
    #im.getbbox() #left up right lower
    im2=im.crop((100, 40, 220, 200))
    print(im2.size)
    im2.save("captured_image.png")

def ml(imgData):
    from PIL import Image
    from keras.models import load_model
    import tensorflow as tf
    print("Processing")
    # a = cv2.imread(imgData)
    # i = np.arange(28*28).reshape(28, 28)
    # imgData = np.fromstring(imgData, dtype=np.float32)
    # print(type(imgData))
    # print(imgData.shape)
    # print(imgData)
    # a = np.expand_dims(imgData, axis=0)
    # test_img = np.expand_dims(a, axis=0)
    # test_img = test_img.transpose((1, 2, 3,0))
    # print(test_img.shape)

    imgstr = re.search(r'base64,(.*)', str(imgData)).group(1)
    with open('captured_image.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))


    # im=Image.open("captured_image.png").convert('L')
    # # print(im.size) #320, 240
    # #im.getbbox() #left up right lower
    # im2=im.crop((100, 40, 220, 200))
    # print(im2.size)
    # im2.save("captured_image.png")

    im = cv2.imread('captured_image.png', 0)[35:200, 108:218]
    # cv2.imshow('img', im)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    im = cv2.resize(im, (28, 28))

    im = im.reshape(1,28,28,1) / 255
    graph = tf.get_default_graph()
    new_model = load_model('mnist_model_5_epochs')
    with graph.as_default():
	    result  = new_model.predict_classes(im)
	    print("Result: ",result)
	    fresult =  str(result[0])

	    print("Final Result",fresult)
	    return fresult
if __name__ == "__main__":
    # run the app locally on the given port
    app.run(port=8080,debug=True)