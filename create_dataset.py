import os
import random

#import pandas
from PIL import Image
#import numpy as np
#from skimage.transform import resize
import csv
import dataset_tool


def pd_to_list(pd,base,label):
    counter = 0
    dataset = []
    for img in pd[0]:
        image = Image.open(os.path.join(base, img))
        image = np.array(image)
        #image = resize(image, (1024, 1024))
        dataset.append([[image, label]])
        counter += 1
        print(str(counter) + "of " + str(pd.shape[0]))
    return dataset


def create_training_set():
    clienttrainfile = r'C:\Users\Simon Smeets\Documents\Unief\2deMaster\Thesis\databases\NUAA\Detectedface\client_train_face.txt'
    impostertrainfile = r'C:\Users\Simon Smeets\Documents\Unief\2deMaster\Thesis\databases\NUAA\Detectedface\imposter_train_face.txt'
    baseclientpath = r'C:\Users\Simon Smeets\Documents\Unief\2deMaster\Thesis\databases\NUAA\Detectedface\ClientFace'
    baseimposterpath = r'C:\Users\Simon Smeets\Documents\Unief\2deMaster\Thesis\databases\NUAA\Detectedface\ImposterFace'
    pdclient = pandas.read_csv(clienttrainfile, " ", header=None)
    pdimposter = pandas.read_csv(impostertrainfile, " ", header=None)

   # nbChunks = (pdclient.shape[0] + pdimposter.shape[0])//200

    #client label == 0
    train_client_images = pd_to_list(pdclient,baseclientpath,0)
    #imposter label == 1
    train_imposter_images = pd_to_list(pdimposter,baseimposterpath,1)

    all_training_images = train_client_images + train_imposter_images
    random.shuffle(all_training_images)
    print('start writing')
    with open(r'C:\Users\Simon Smeets\Documents\Unief\2deMaster\Thesis\databases\NUAA\Detectedface\full_training_dataset_lq', 'w') as file:
        filewriter = csv.writer(file, delimiter=',')
        filewriter.writerows(all_training_images)


def create_test_set():
    clienttestfile = r'C:\Users\Simon Smeets\Documents\Unief\2deMaster\Thesis\databases\NUAA\Detectedface\client_test_face.txt'
    impostertestfile = r'C:\Users\Simon Smeets\Documents\Unief\2deMaster\Thesis\databases\NUAA\Detectedface\imposter_test_face.txt'
    baseclientpath = r'C:\Users\Simon Smeets\Documents\Unief\2deMaster\Thesis\databases\NUAA\Detectedface\ClientFace'
    baseimposterpath = r'C:\Users\Simon Smeets\Documents\Unief\2deMaster\Thesis\databases\NUAA\Detectedface\ImposterFace'
    pdclient = pandas.read_csv(clienttestfile, " ", header=None)
    pdimposter = pandas.read_csv(impostertestfile, " ", header=None)

   # nbChunks = (pdclient.shape[0] + pdimposter.shape[0])//200

    #client label == 0
    test_client_images = pd_to_list(pdclient,baseclientpath,0)
    #imposter label == 1
    test_imposter_images = pd_to_list(pdimposter,baseimposterpath,1)

    all_test_images = test_client_images + test_imposter_images
    random.shuffle(all_test_images)
    print('start writing')
    with open(r'C:\Users\Simon Smeets\Documents\Unief\2deMaster\Thesis\databases\NUAA\Detectedface\full_test_dataset_lq', 'w') as file:
        filewriter = csv.writer(file, delimiter=',')
        filewriter.writerows(all_test_images)

tfrecord_dir = "../tfFFHQ"
image_dir = "../databases/FFHQ/thumbnails128x128/*"
shuffle = True
dataset_tool.create_from_images(tfrecord_dir, image_dir, shuffle)
