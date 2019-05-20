import numpy as np
from scipy import signal as sg

array = np.float32(np.random.rand(1000,1000))
kernel = np.random.rand(3,3)
f1 = open("pyarray.in", "w")
header = "b f32"
f1.write(header)
f1.write(array)
f1.write(kernel)
f1.close()

result = sg.convolve2d(array,kernel)
f2 = open("pyresult.in", "w")
f2.write(header)
f2.write(result)
f2.close()