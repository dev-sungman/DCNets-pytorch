from random import *
import os.path

def visualize_2d(data_path, num, batch_size, model, feature_size):

    # 0. select classes for visualize
    sample_class = [int(i) for i in range(num)]
    
    # 1. load data
    data_path = os.path.abspath(data_path)
    class_list = os.listdir(data_path)
    
    # calculate
    file_num = 0
    for i in sample_class:
        file_num += len(os.listdir(data_path + '/' + class_list[i]))


    embeddings = np.zeros([file_num, feature_size])
    

    # 2. extract features

    # 3. visualization using t-sne



if __name__ == '__main__':
    visualize_2d('../datasets/faces_emore/imgs', 2, 256, 1, 512)
