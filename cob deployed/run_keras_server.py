# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
# from keras.applications import ResNet50
from flask import Flask,redirect,url_for,render_template,request,jsonify
import requests,os
app = Flask(__name__)
from PIL import Image
import json
import numpy as np
import cv2
import tensorflow as tf
from flask import send_file
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()

# some fixes 
graph=None
graph = tf.get_default_graph()

loaded_model = model_from_json(loaded_model_json)
#load woeights into new model
loaded_model.load_weights("ReTraining.h5")
print("Loaded Model from disk")

@app.route('/about',methods=['GET'])
def results():
	return jsonify("U are in home page, there is nothing here, go for /post and upload image")

def weAreLive(url):
	pic_url=url
	with open('test.jpg', 'wb') as handle:
	        response = requests.get(pic_url, stream=True)

	        if not response.ok:
	            print (response)

	        for block in response.iter_content(1024):
	            if not block:
	                break

	            handle.write(block)
	#     preprocessing the image 
	image = cv2.imread('test.jpg')
	image = cv2.resize(image, (160,160))
	image = img_to_array(image)
	image/=255
	#     cv2.imshow('ImageWindow',image)
	#     cv2.waitKey()
	#         Making the predictions
	img=np.expand_dims(image, axis=0)
	# clear_session()
	global loaded_model
	# loaded_model._make_predict_function()
	with graph.as_default():
		res=loaded_model.predict(img)
		print(img)
		pass
	os.remove("test.jpg")
	# if res[0][0]>res[0][1]:
	#     return "Baseball"
	# else:
	#     return "Cricket"
	return "erros from preds"
	#     deleting the file at last


# request would be made from js code
@app.route("/predict",methods=["POST"])
def predict():
	# what if user didnot apply any filter
	x=request.form.get("url")
	print(x)
	x="https://i.ytimg.com/vi/O5bV5LVVr5A/maxresdefault.jpg"
	res=weAreLive(x)
	return json.dumps({"res":res})
	# return json.dumps({"res":"scsasa"})

if __name__ == '__main__':
	app.debug=True
	app.run()