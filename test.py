import numpy as np
import time

t=time.time()

m=np.random.rand(2000,3000)
n=np.random.rand(2000,3000)
x=m+n

print(time.time()-t)