import numpy as np

def position_agent(image):
    for i in range(image.shape[1]):
        a = np.sum(image[6,i,:],0) 
        if a == 232 : #color agent
            #position x y exacte : (191,i+3)
            return i+3
    return 7000
        
def position_alien(image):
    j = 0
    while True:
        for i in range(image.shape[1]):
            a = np.sum(image[j,i,:],0) 
            if a == 297 : ## color alien
                False
                return i+3
        if j<image.shape[0]-1:
            j +=1
        else:
            plt.figure()
            plt.imshow(image)
            return 7000


