from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath("model_ex-050_acc-0.995781.h5")
prediction.setJsonPath("idenprof_model_class.json")
prediction.loadModel(num_objects=5)

predictions, probabilities = prediction.predictImage("image5.jpg", result_count=3)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)