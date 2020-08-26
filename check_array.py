import numpy as np
a = np.loadtxt('labels1.out').reshape(1,9,13,13)
b = np.loadtxt('labels2.out').reshape(1,9,13,13)
print(np.allclose(a,b))