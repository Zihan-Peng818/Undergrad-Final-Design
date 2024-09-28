import numpy
import numpy as np
import os

label_path = "VL-CMU-CD/labels"
index = os.listdir(label_path)
result = 0
count = 0
for i in index:
    labels = os.listdir(os.path.join(label_path, i))
    if i == "121":
        break
    for item in labels:
        label = np.load(os.path.join(label_path, i, item))
        label[label <= 1] = 0
        label[label > 1] = 1
        positive = numpy.sum(label)
        b = positive/786432
        print(b)
        result += positive / 786432
        count += 1
print(count)
print(result)
result = result/count
print(result)

