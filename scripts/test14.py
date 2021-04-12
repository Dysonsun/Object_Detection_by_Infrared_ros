import numpy as np

data = np.array([1,2,3,4,5,6,7,8])
print(data)
data = data.reshape(2,2,2)
print(data)
data = np.tile(data,(1,1,1,1))
print("shape",data.shape)
print(data)