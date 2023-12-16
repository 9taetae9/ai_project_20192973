from roboflow import Roboflow
import torch

rf = Roboflow(api_key="gCCzibGmqfTChcQtkUeZ")
project = rf.workspace().project("namsan-tower-detection")
model = project.version(1).model

torch.save(model, 'namsan.pt')

# infer on a local image
#print(model.predict("your_image.jpg", confidence=40, overlap=30).json())

# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())