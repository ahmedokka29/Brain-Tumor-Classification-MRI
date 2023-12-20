#!/usr/bin/env python
# coding: utf-8

import os
import grpc

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from base import BasePreprocessor

from flask import Flask
from flask import request
from flask import jsonify

from proto import np_to_protobuf

host = os.getenv('TF_SERVING_HOST', 'localhost:8500')

channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)


preprocessor = BasePreprocessor(target_size=(224, 224))


def prepare_request(X):
    pb_request = predict_pb2.PredictRequest()

    pb_request.model_spec.name = 'brain-cancer-model'
    pb_request.model_spec.signature_name = 'serving_default'

    pb_request.inputs['efficientnetv2-b3_input'].CopyFrom(np_to_protobuf(X))
    return pb_request


classes = ['glioma_tumor',
           'meningioma_tumor',
           'no_tumor',
           'pituitary_tumor'
           ]



def prepare_response(pb_response):
    preds = pb_response.outputs['dense_3'].float_val
    return dict(zip(classes, preds))


def predict(url):
    X = preprocessor.from_url(url)
    pb_request = prepare_request(X)
    pb_response = stub.Predict(pb_request, timeout=30.0)
    response = prepare_response(pb_response)
    return response


app = Flask('gateway')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    url = data['url']
    result = predict(url)
    return jsonify(result)


if __name__ == '__main__':
    # url = 'https://raw.githubusercontent.com/ahmedokka29/ml-zoomcamp/main/11_capstone_1/data/Testing/glioma_tumor/image(1).jpg'
    # response = predict(url)
    # print(response)
    app.run(debug=True, host='0.0.0.0', port=9696)
