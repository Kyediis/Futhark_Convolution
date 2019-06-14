import timeit as t
# code snippet to be executed only once 
mysetup = ''' 
from scipy import signal as sg
import numpy as np
kernel = np.random.rand(3,3)
arr512 = np.float32(np.random.rand(512,512))
arr1024 = np.float32(np.random.rand(1024,1024))
arr2048 = np.float32(np.random.rand(2048,2048))
'''
  
# code snippet whose execution time is to be measured 
mycode512 = '''
sg.convolve2d(arr512,kernel)
'''

mycode1024 = ''' 
sg.convolve2d(arr1024,kernel)
'''

mycode2048 = ''' 
sg.convolve2d(arr2048,kernel)
'''
  
# timeit statement 
print t.timeit(setup = mysetup, stmt = mycode512, number = 1)
print t.timeit(setup = mysetup, stmt = mycode1024, number = 1) 
print t.timeit(setup = mysetup, stmt = mycode2048, number = 1)  

