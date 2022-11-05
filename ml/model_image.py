######### 가상환경 - py38 #########

import cv2
import numpy as np #배열 계산 용이
from PIL import Image #python imaging library
import os

visit_list = []  # 방문완료한 사용자

########################## 데이터셋 만드는 부분 #################################
# 이후 변경 - DB의 이미지로 학습이 이루어지도록
def getImagesAndLabels(path):
    detector = cv2.CascadeClassifier(r"C:\Users\Yewon\anaconda3\envs\py38\Library\etc\haarcascades\haarcascade_frontalface_default.xml")

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # listdir : 해당 디렉토리 내 파일 리스트
    # path + file Name : 경로 list 만들기

    faceSamples = []
    ids = []
    for imagePath in imagePaths:  # 각 파일마다
        # 흑백 변환
        PIL_img = Image.open(imagePath).convert('L')  # L : 8 bit pixel, bw
        img_numpy = np.array(PIL_img, 'uint8')

        # user id
        id = int(os.path.split(imagePath)[-1].split(".")[1])  # 마지막 index : -1

        # 얼굴 샘플
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return faceSamples, ids


########################## 영상에서 얼굴 검출 #################################
def face_recognition():
    #classifier
    faceCascade = cv2.CascadeClassifier(r"C:\Users\Yewon\anaconda3\envs\py38\Library\etc\haarcascades\haarcascade_frontalface_default.xml")

    #video caputure setting
    # capture = cv2.VideoCapture(0)   # initialize, # is camera number
    capture = cv2.imread("./김예원_흰배경.jpg")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280) #CAP_PROP_FRAME_WIDTH == 3
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT,720) #CAP_PROP_FRAME_HEIGHT == 4
    # 정상작동시 true

    #console message  - int형으로 입력받음
    names = []

    face_id = input('\n enter user id end press <return> ==> ')
    names.append(face_id)    # names를 append 하게 함
    print("\n [INFO] Initializing face capture. Look the camera and wait ...")

    count = 0 # # of caputre face images
    #영상 처리 및 출력
    while True:
        ret, frame = capture.read() #카메라 상태 및 프레임
        # cf. frame = cv2.flip(frame, -1) 상하반전
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 흑백으로
        faces = faceCascade.detectMultiScale(          # 얼굴 위치 검출
            gray,#검출하고자 하는 원본이미지
            scaleFactor = 1.2, #검색 윈도우 확대 비율, 1보다 커야 한다
            minNeighbors = 6, #얼굴 사이 최소 간격(픽셀)
            minSize=(20,20) #얼굴 최소 크기. 이것보다 작으면 무시
        )

        #얼굴에 대해 rectangle 출력
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            #inputOutputArray, point1 , 2, colorBGR, thickness)
            count += 1
            cv2.imwrite(".././dataset/User."+str(face_id)+'.'+str(count)+".jpg",gray[y:y+h, x:x+w])

        cv2.imshow('image', frame)

        #종료조건
        if cv2.waitKey(1) > 0 : break #키 입력이 있을 때 반복문 종료
        elif count >= 100 : break #100 face sample



    print("\n [INFO] Exiting Program and cleanup stuff")
    print(names)

    capture.release() #메모리 해제
    cv2.destroyAllWindows()#모든 윈도우 창 닫기



########################## 데이터셋 바탕으로 학습 #################################

face_recognition()

path = '.././dataset' #경로 (dataset 폴더)
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(r"C:\Users\Yewon\anaconda3\envs\py38\Library\etc\haarcascades\haarcascade_frontalface_default.xml")


print('\n [INFO] Training faces. It will take a few seconds. Wait ...')
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids)) #학습

# recognizer.write('.././trainer/trainer.yml')
recognizer.write('.././trainer/trainer.pkl')  #!!
print('\n [INFO] {0} faces trained. Exiting Program'.format(len(np.unique(ids))))


cv2.destroyAllWindows()


########################## 모델 저장 #################################
# # 피클 사용 - https://guru.tistory.com/40

visit_list = list(set(visit_list))


