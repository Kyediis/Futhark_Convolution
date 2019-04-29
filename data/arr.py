import numpy as np
from scipy import signal as sg

array = np.random.rand(1000,1000)
kernel = np.random.rand(3,3)
f1 = open("data/pyarray.txt", "w")
f1.write(array)
f1.write(kernel)
f1.close()

result = sg.convolve2d(array,kernel)
f2 = open("data/pyresult.txt", "w")
f2.write(result)
f2.close()