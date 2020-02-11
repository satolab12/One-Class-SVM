import numpy as np
import cv2
from skimage.feature import hog
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import pickle

n = 3
n_dim = 4
alpha = - 1.0e+6
th = 20 #3
nu = 0.3 #0.1 #入力データの異常値の割合
font = cv2.FONT_HERSHEY_COMPLEX
train_data = "./dataset/train/train.csv"
weights = "./dataset/weights/weights.sav"
weights_pca = "./dataset/weights/weights_pca.sav"

f_ = cv2.CascadeClassifier()  # "./cascades/haarcascade_fullbody.xml"
f_.load(cv2.samples.findFile("./cascades/haarcascade_frontalface_alt.xml"))

def preprocess(image):
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame = cv2.equalizeHist(frame)
    return frame

def data_collect():
    feature = []
    capture = cv2.VideoCapture(0)

    while (True):
        ret, frame = capture.read()
        frame = preprocess(frame)
        face = f_.detectMultiScale(frame)  # ,scaleFactor=1.2

        for rect in face:
            cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), (255, 255, 0), thickness=2)
            face_frame = frame[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
            face_frame = cv2.resize(face_frame, (60, 60))
            hog_f_, im = hog(face_frame, visualise=True,transform_sqrt=True)
            feature = np.append(feature,hog_f_)
            np.savetxt(train_data,feature.reshape(-1,2025), delimiter=",")
            cv2.putText(frame, "please smile for collecting data!", (10, 100), font,
                             1, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.waitKey(1)
        cv2.imshow("face", frame)

def train():
    x_train = np.loadtxt(train_data,delimiter=",")
    pca = PCA(n_components=n_dim)
    clf = OneClassSVM(nu=nu, gamma=50/n_dim)#1/n_dim
    z_train = pca.fit_transform(x_train)
    clf.fit(z_train)

    pickle.dump(pca, open(weights_pca, "wb"))
    pickle.dump(clf,open(weights,"wb"))

def main():
    clf = pickle.load(open(weights,"rb"))
    pca = pickle.load(open(weights_pca, "rb"))
    capture = cv2.VideoCapture(0)

    while(True):
        ret,frame = capture.read()
        frame = preprocess(frame)
        face = f_.detectMultiScale(frame)

        for rect in face:
            cv2.rectangle(frame,tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]),(255,255,0),thickness=2)
            face_frame = frame[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            face_frame = cv2.resize(face_frame,(60,60))
            feature , _ = hog(face_frame,visualise=True,transform_sqrt=True)
            z_feature = pca.transform(feature.reshape(1,2025))
            score = clf.predict(z_feature.reshape(1,n_dim))
            if score[0]== 1:
                cv2.putText(frame, "smile!", (10, 100), font,
                             1, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.waitKey(1)
        cv2.imshow("face",frame)#まさかのv,uで指定

if __name__ == '__main__':
    #data_collect()
    train()
    main()
