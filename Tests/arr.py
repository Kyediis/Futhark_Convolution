import binascii
import numpy as np
from scipy import signal as sg



#construct random data values
array = np.float32(np.random.rand(4,4))

f1 = open("pyarray.in", "wb")
#create header containing (binary, dimensions, type, 64bit number of elements)

header1 = "b f32"+binascii.unhexlify("04000000000000000400000000000000")
#header1 = "b f32"+binascii.unhexlify("00040000000000000004000000000000")
f1.write(header1)
f1.write(array)

#same as above but with different values for kernel
kernel = np.float32(np.asarray([[0.0,-1.0,0.0],[-1.0,5.0,-1.0],[0.0,-1.0,0.0]]))
header2 = "b f32"+binascii.unhexlify("03000000000000000300000000000000")
f1.write(header2)
f1.write(kernel)
f1.close()

#calculate result and write same header as original data
result = sg.convolve2d(array, kernel, 'same')
f2 = open("pyresult.out", "wb")
f2.write(header1)
f2.write(result)
f2.close()
