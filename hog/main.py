import os
import numpy as np


from skimage.feature import hog
from skimage import color, io
from skimage import transform

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.externals import joblib

from sklearn.cross_validation import train_test_split
from sklearn import metrics
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
# 从路径中读取folio图片
def read_folio_images(path):
    names_dir=os.listdir(path)
    names=[]
    images=[]
    for name in names_dir:
        pics=os.listdir(path+name)
        for pic in pics:
            names.append(name)
            images.append(io.imread(path+name+'/'+pic))
        print("reading :"+name+"   sucessful!!")

    print("ALL reading is Done!!")

    return names,images

def change_save_images(i_path, o_path):
    names_dir = os.listdir(i_path)
    names = []
    images = []
    kind=0
    for name in names_dir:
        if not os.path.exists(o_path+'/'+name):
            os.makedirs(o_path + '/' + name)
        pics = os.listdir(i_path + name)
        temp=0
        kind+=1
        for pic in pics:
            temp+=1
            image=io.imread(i_path + name + '/' + pic)
            if len(image)>len(image[0]):
                image = transform.resize(image, (688, 387))
            else :
                image = transform.resize(image, (387, 688))
            io.imsave(o_path+'/'+name+'/'+pic,image)
            print(str(temp)+"/"+str(kind))

#从单张图片中获取HOG的feature
def HOG_feature(image):
    image = color.rgb2gray(image)
    if len(image) > len(image[0]):
        image = transform.resize(image, (688, 387))
    else:
        image = transform.resize(image, (387, 688))
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
    print("Step 2 : get hog features:")
    total = len(images)
    current =0
    for image in images:
        current+=1
        image_gray = color.rgb2gray(image)
        if len(image) > len(image[0]):
            image_gray_resize = transform.resize(image_gray, (688, 387))
        else:
            image_gray_resize = transform.resize(image_gray, (387, 688))
        HOG_features.append(
            hog(image_gray_resize,
                orientations=9,
                pixels_per_cell=(8,8),
                cells_per_block=(3,3))
        )
        print(str(current)+'/'+str(total)+" : proceed")
    images.clear()
    print("Step 2: get hog features successful")
    return HOG_features

def SVM_train(names,HOG_features):
    print("Step 4:Start training!!")
    clf=svm.LinearSVC()
    X = np.array(HOG_features)
    y = np.array(names)
    clf.fit(X,y)
    joblib.dump(clf,'clf.model')
    print("Step 4:Training scussfull!!")
    return clf

def folio_SVM_Testing(names,HOG_features):
    X = np.array(HOG_features)
    y = np.array(names)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)

    clf = svm.LinearSVC()
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    print (metrics.accuracy_score(y_test,y_pred))


def PCA_low(Hog_features):
    print("Step 3:PCA starting")
    pca = PCA(n_components=0.9)
    pca.fit(Hog_features)
    joblib.dump(pca, 'pca.model')
    print("Step 3:PCA sucessful!!")
    return pca.transform(Hog_features)

def prediction(image_path):
    clf = joblib.load('clf.model')
    pca = joblib.load('pca.model')
    test_image=io.imread(image_path)
    if len(test_image) > len(test_image[0]):
        test_image = transform.resize(test_image, (688, 387))
    else:
        test_image = transform.resize(test_image, (387, 688))
    #test_image=transform.resize(test_image,(387,688))
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
    #HOG_features = PCA_low(HOG_features)
    clf = SVM_train(labels, HOG_features)

def train_folio():
    names, images = read_folio_images("../train_picture_folio_387_688/")
    HOG_features=get_HOG_features(images)
    HOG_features=PCA_low(HOG_features)
    clf=folio_SVM_Testing(names,HOG_features)

if __name__ == '__main__':
    train_folio()
    # result = prediction('../train_pictures/pictures/斑叶堇菜叶反.bmp')
    #
    # labels_names_path = "../train_pictures/label_names.txt"
    # dic = read_dict_label_names(labels_names_path)
    # result = prediction('E:\\Project\\Pycharm\\HOG_SVM_py\\train_picutre_folio\\caricature plant\\20150410_154744.jpg')
    # #
    # print("预测结果是："+result[0])
    #change_save_images("../train_picutre_folio/", "../train_picture_folio_387_688/")

