from cv2 import medianBlur
from matplotlib import pyplot as plt
import numpy as np

def PhaseReduce(input, xs, ys, EDGE):
    xs = np.trunc(np.array(xs)).astype(int)
    ys = np.trunc(ys).astype(int)
    layers = np.ones((input.shape[0], input.shape[1])) * -1
    maxlayer = len(xs)
    neighs = np.zeros((8, 2))
    queue = []
    
    for i in range(0, maxlayer):
        queue.append([xs[i], ys[i]])
        layers[xs[i], ys[i]] = i

        while queue != []:
            [x, y] = queue.pop(0)

            neighs[0, :] = [x-1, y-1]
            neighs[1, :] = [x-1, y  ]
            neighs[2, :] = [x-1, y+1]
            neighs[3, :] = [x  , y+1]      
            neighs[4, :] = [x+1, y+1]
            neighs[5, :] = [x+1, y  ]
            neighs[6, :] = [x+1, y-1]
            neighs[7, :] = [x  , y-1]

            neighs = neighs.astype(int)

            for j in range(0, 8):

                if layers[neighs[j, 0], neighs[j, 1]] != -1:
                    pass
                elif EDGE[neighs[j, 0], neighs[j, 1]]:
                    layers[neighs[j, 0], neighs[j, 1]] = i
                else:
                    layers[neighs[j, 0], neighs[j, 1]] = i
                    queue.append([neighs[j, 0], neighs[j, 1]])
    
    input = np.uint16(input) + layers*255
    input = np.uint8((input - input.min())/(input.max() - input.min())*255)
    for i in range(3):
        input = medianBlur(input, 3)

    return input
                

