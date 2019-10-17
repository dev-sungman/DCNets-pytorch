from PIL import Image
import numpy as np
import cv2
import os

class Visualizer:
    def __init__(self, sample_class_num, sample_num, model, embedding_size, random_class=False):
        self.sample_class_num = sample_class_num
        self.data = np.zeros([(sample_class_num*sample_num), (embedding_size+1)])
        self.idx = 0

        # select classes to visualize
        # random_class : if random_class is False, classes will be selected sequentially

        if random_class is True:
            #TODO: random sampling for select classes
            self.sample_class = 1
        else:
            self.sample_class = [int(i) for i in range(sample_class_num)]
    
    def set_sample(self, labels, embeddings):
        for i in range(len(labels)):
            c = labels[i].numpy()
            if c in self.sample_class:
                self.data[self.idx,0] = c
                self.data[self.idx,1:] = embeddings[i,:]
            self.idx += len(labels)

            if self.idx > self.data.shape[0]:


        print('data setting is finished')
        print(self.data)
