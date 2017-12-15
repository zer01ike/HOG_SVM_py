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

import  multiprocessing

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
        #print("reading :"+name+"   sucessful!!")

    #print("ALL reading is Done!!")

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
def HOG_feature(image,o,i,j):
    image = color.rgb2gray(image)
    if len(image) > len(image[0]):
        image = transform.resize(image, (688, 387))
    else:
        image = transform.resize(image, (387, 688))
    result = hog(image,
                orientations=o,
                pixels_per_cell=(i,i),
                cells_per_block=(j,j),
                block_norm='L2-Hys',)
    return result

#从文件目录中的所有文件获取feature
def get_HOG_features(images,o,i,j):
    #Attention: 图片的大小必须一致
    HOG_features=[]
    #print("Step 2 : get hog features:")
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
                orientations=o,
                pixels_per_cell=(i,i),
                cells_per_block=(j,j))
        )
        #print(str(current)+'/'+str(total)+" : proceed")
    #images.clear()
    #print("Step 2: get hog features successful")
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
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

    clf = svm.LinearSVC()
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    return metrics.accuracy_score(y_test,y_pred)


def PCA_low(Hog_features):
    #print("Step 3:PCA starting")
    pca = PCA(n_components=0.9)
    pca.fit(Hog_features)
    #joblib.dump(pca, 'pca.model')
    #print("Step 3:PCA sucessful!!")
    return pca.transform(Hog_features)

def prediction(image_path):
    clf = joblib.load('clf.model')
    pca = joblib.load('pca.model')
    test_image=io.imread(image_path)
    if len(test_image) > len(test_image[0]):
        test_image = transform.resize(test_image, (688, 387))
    else:
        test_image = transform.resize(test_image, (387, 688))
    io.imsave("test.jpg",test_image)
    #test_image=transform.resize(test_image,(387,688))
    test_image=color.rgb2gray(test_image)

    hog_features=HOG_feature(test_image,10,38,3)
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

def train_folio_find_param():
    names, images = read_folio_images("../train_picture_folio_387_688/")
    file=open("../result0.2.txt",'a')
    max_o=0
    max_i=0
    max_j=0
    for o in range(5,15):
        max_o = o
        for i in range(20, 40):
            max_i = i
            max_clf=0
            for j in range(3, 7):
                HOG_features = get_HOG_features(images,o, i, j)
                HOG_features = PCA_low(HOG_features)
                clf = folio_SVM_Testing(names, HOG_features)
                if clf > max_clf:
                    max_clf= clf
                    max_j = j
            print(str(max_o) + "+"+str(max_i) + "+" + str(max_j) + "=" + str(max_clf))
            file.writelines(str(max_o) + " "+str(max_i) + " " + str(max_j) + " " + str(max_clf)+"\n")
    file.close()

def train_folio():
    names, images = read_folio_images("../train_picture_folio_387_688/")
    HOG_features = get_HOG_features(images,10,38,3)
    HOG_features = PCA_low(HOG_features)
    clf = SVM_train(names,HOG_features)

def multipro_train_folio_find_param(o,i,j):
    names, images = read_folio_images("../train_picture_folio_387_688/")
    HOG_features = get_HOG_features(images,o, i, j)
    HOG_features = PCA_low(HOG_features)
    clf = folio_SVM_Testing(names, HOG_features)
    #resultq.put(str(param[0]) + "+"+str(param[1]) + "+" + str(param[2]) + "=" + str(clf))
    print(str(o) + "+"+str(i) + "+" + str(j) + "=" + str(clf))

if __name__ == '__main__':
    cores=multiprocessing.cpu_count()
    pool=multiprocessing.Pool(processes=cores)

    #result= pool.apply_async(multipro_train_folio_find_param,args=)

    taskq = [(o,i,j) for o in range(5,15) for i in range(20,40) for j in range (3,6)]
    result = pool.starmap_async(multipro_train_folio_find_param, taskq)
    print(result.get())
    pool.close()
    pool.join()

    # taskq=multiprocessing.Queue()
    # resultq=multiprocessing.Queue()
    # clock=multiprocessing.Lock()
    #
    # p=multiprocessing.Process(target=multipro_train_folio_find_param,args=(taskq,clock))
    # p.start()
    # for o in range(5,15):
    #     for i in range(20,40):
    #         for j in range(3,6):
    #             taskq.put([o,i,j])
    # taskq.close()
    #
    # #print(resultq.get())
    # p.terminate()
    #
    # p.join()


    #train_folio_find_param()
    #train_folio()
    # result = prediction('../train_pictures/pictures/斑叶堇菜叶反.bmp')
    #
    # labels_names_path = "../train_pictures/label_names.txt"
    # dic = read_dict_label_names(labels_names_path)
    # result = prediction('E:\\Project\\Pycharm\\HOG_SVM_py\\train_picutre_folio\\caricature plant\\20150410_154744.jpg')
    # #
    # print("预测结果是："+result[0])
    #change_save_images("../train_picutre_folio/", "../train_picture_folio_387_688/")
    # result = prediction("../train_picture_folio_387_688/papaya/20150524_183008.jpg")
    # print(result)
