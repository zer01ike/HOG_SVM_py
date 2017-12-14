import os
import numpy as np


from skimage.feature import hog
from skimage import color ,io
from skimage import transform

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.externals import joblib



def read_train_images(path):
    #打开文件目录，获取里面所有的图像文件添加到一个list中。
    pics=os.listdir(path)
    images=[]
    for pic in pics:
        images.append(io.imread(path+pic))

    return images
def read_train_labels(labels_path):
    label_file=open(labels_path)
    labels=[]

    for label in label_file.readlines():
        labels.append(int(label))
    return labels

def read_dict_label_names(label_name_path):
    label_name_file= open(label_name_path)
    labels=[]
    names=[]
    for lable_name in label_name_file.readlines():
        temp=lable_name.split(" ")
        #print(lable_name)
        labels.append(int(temp[0]))
        names.append(temp[1])
    return dict(zip(labels, names))

#从单张图片中获取HOG的feature
def HOG_feature(image):
    image = color.rgb2gray(image)
    image = transform.resize(image,(128,128))
    result = hog(image,
                orientations=9,
                pixels_per_cell=(8,8),
                cells_per_block=(3,3),
                block_norm='L2-Hys',)
    return result

#从文件目录中的所有文件获取feature
def get_HOG_features(images):
    #Attention: 图片的大小必须一致
    HOG_features=[]
    for image in images:
        image_gray = color.rgb2gray(image)
        image_gray_resize = transform.resize(image_gray,(128,128))
        HOG_features.append(
            hog(image_gray_resize,
                orientations=9,
                pixels_per_cell=(8,8),
                cells_per_block=(3,3))
        )
    return HOG_features

def SVM_train(names,HOG_features):
    clf=svm.LinearSVC()
    X = np.array(HOG_features)
    y = np.array(names)
    clf.fit(X,y)
    joblib.dump(clf,'clf.model')
    return clf

def PCA_low(Hog_features):
    pca = PCA(n_components=0.9)
    pca.fit(Hog_features)
    joblib.dump(pca, 'pca.model')
    return pca.transform(Hog_features)

def prediction(image_path):
    clf = joblib.load('clf.model')
    pca = joblib.load('pca.model')
    test_image=io.imread(image_path)
    test_image=transform.resize(test_image,(128,128))
    test_image=color.rgb2gray(test_image)

    hog_features=HOG_feature(test_image)
    hog_features=pca.transform([hog_features])

    return clf.predict(hog_features)
def train():
    images_path = "../train_pictures/pictures/"
    images = read_train_images(images_path)
    labels_path="../train_pictures/label.txt"
    labels=read_train_labels(labels_path)

    labels_names_path="../train_pictures/label_names.txt"


    HOG_features = get_HOG_features(images)
    HOG_features = PCA_low(HOG_features)
    clf = SVM_train(labels, HOG_features)

if __name__ == '__main__':
    #train()
    result = prediction('../train_pictures/pictures/斑叶堇菜叶反.bmp')

    labels_names_path = "../train_pictures/label_names.txt"
    dic = read_dict_label_names(labels_names_path)
    #result = prediction('/Users/zer0like/Downloads/300.jpeg')

    print("预测结果是："+dic[result[0]])