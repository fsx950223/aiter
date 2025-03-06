from ctypes import *
import os


lib = CDLL('./math_test.so')
res = c_int(0)
lib.call.argtypes = [c_int, c_int, POINTER(c_int)] 
lib.call(1, 2, res)
print(res.value)