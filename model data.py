from keras.models import Sequential
from keras.models import load_model

model = load_model("./model-018.model")  # loads in model
for layer in model.layers:
    print("The Array is: ", layer.get_weights() )  # printing the array
# This prints out the information on the layers
model.summary()
