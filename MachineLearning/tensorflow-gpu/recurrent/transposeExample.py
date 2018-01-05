import numpy as np

x = np.ones((1,2,3))

print(x)
print(np.transpose(x, (1,0,2)))

'''
TRANSPOSE BECAUSE THAT IS THE DATA STRUCTURE IT NEEDS

[[   
    [ 1.  1.  1.],
    [ 1.  1.  1.]   
]]
[  
    [[ 1.  1.  1.]],
    [[ 1.  1.  1.]] 
]
 
'''