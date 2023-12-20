FROM tensorflow/serving:2.7.0

COPY brain-cancer-model /models/brain-cancer-model/1

ENV MODEL_NAME="brain-cancer-model"
