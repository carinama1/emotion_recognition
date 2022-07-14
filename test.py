
import json
import cv2
import numpy as np

print(np.__version__)

print(cv2.__version__)
# Data to be written
dictionary = {"emotions": []}

x = 19
dictionary["tahu"] = "bulat"
dictionary[x] = "Test"
dictionary["emotions"].append("test")
# Serializing json
json_object = json.dumps(dictionary, indent=4)

# Writing to sample.json
with open("sample.json", "w") as outfile:
    outfile.write(json_object)
