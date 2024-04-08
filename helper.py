import numpy as np
import pandas as pd
from PIL import Image

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((32, 32))
    return image

# df = pd.read_csv("cifar-10/trainLabels.csv")
# labels = df['label'].values

images = []
for i in range(30000):
    image_path = f"cifar-10/test/{i+1}.png"
    image = preprocess_image(image_path)
    images.append(np.array(image))

X_test = np.array(images).reshape((30000, -1)).T
# Y_train = np.array(labels)

np.save("test.npy", X_test)
# np.save("train_labels.npy", Y_train)
