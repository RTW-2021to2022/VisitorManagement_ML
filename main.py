# https://wings2pc.tistory.com/entry/%EC%9B%B9-%EC%95%B1%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%ED%94%8C%EB%9D%BC%EC%8A%A4%ED%81%ACPython-Flask?category=777829
# https://go-guma.tistory.com/9
import flask
from flask import Flask, request, render_template
import cv2
from datetime import datetime
import os
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename


app = Flask(__name__)
# 메인페이지 - url 요청시 기본 index.html로 이동 (렌더링)
@app.route("/", methods=['POST','GET'])
def cam_main():
    return render_template('camera.html')

@app.route("/camera", methods=['POST','GET'])
def cam_main2():
    return render_template('camera.html')


@app.route("/update", methods=['POST', 'GET'])    # 변경가능, 임시
def retrain():
    if request.method == 'POST':
        file = request.files['image']

        faceCascade = cv2.CascadeClassifier(
            r"C:\Users\Yewon\anaconda3\envs\py38\Library\etc\haarcascades\haarcascade_frontalface_default.xml")

        file.save("./"+ secure_filename(f.filename))
        image = cv2.imread(file)  ## file 자체가 아닌 경로를 읽어오기 때문에 오류 발생
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_id = input('\n enter user id end press <return> ==> ')

        # console message  - int형으로 입력받음
        names = []

        count = 0
        # 영상 처리 및 출력
        while True:
            faces = faceCascade.detectMultiScale(  # 얼굴 위치 검출
                gray,  # 검출하고자 하는 원본이미지
                scaleFactor=1.2,  # 검색 윈도우 확대 비율, 1보다 커야 한다
                minNeighbors=6,  # 얼굴 사이 최소 간격(픽셀)
                minSize=(20, 20)  # 얼굴 최소 크기. 이것보다 작으면 무시
            )

            # 얼굴에 대해 rectangle 출력
            for (x, y, w, h) in faces:
                cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1
                cv2.imwrite("./dataset2/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

            # cv2.imshow('image', gray)

            # 종료조건
            if cv2.waitKey(1) > 0:
                break
            elif count >= 100:
                break  # 데이터셋 만드는 수 지정

        print("\n [INFO] Exiting Program and cleanup stuff")

        cv2.destroyAllWindows()  # 모든 윈도우 창 닫기

        #----
        path = './dataset2'  # 경로 (dataset 폴더)
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier(
            r"C:\Users\Yewon\anaconda3\envs\py38\Library\etc\haarcascades\haarcascade_frontalface_default.xml")

        print('\n [INFO] Training faces. It will take a few seconds. Wait ...')

        def getImagesAndLabels(path):
            detector = cv2.CascadeClassifier(
                r"C:\Users\Yewon\anaconda3\envs\py38\Library\etc\haarcascades\haarcascade_frontalface_default.xml")

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

        # ------
        faces, ids = getImagesAndLabels(path)  # 데이터셋에서 뽑아냄
        recognizer.train(faces, np.array(ids))  # 학습

        # 모델 저장
        recognizer.write('./trainer/trainer.pkl')
        print('\n [INFO] {0} faces trained. Exiting Program'.format(len(np.unique(ids))))

        cv2.destroyAllWindows()

        return flask.render_template('camera.html', d0="업데이트를 완료했습니다!")




# 데이터
# 데이터 예측 처리
visit_list=[]
tmp = []

@app.route('/predict', methods=['POST', 'GET'])
def make_prediction():
    if request.method == 'POST':

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('./trainer/trainer.pkl')           # 학습된 모델을 불러옴
        faceCascade = cv2.CascadeClassifier(
            r"C:\Users\Yewon\anaconda3\envs\py38\Library\etc\haarcascades\haarcascade_frontalface_default.xml")

        font = cv2.FONT_HERSHEY_SIMPLEX

        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        minW = 0.1 * cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        minH = 0.1 * cam.get(cv2.CAP_PROP_FRAME_HEIGHT)


        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=6,
                minSize=(int(minW), int(minH))
            )

            now = datetime.now()
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 153, 103), 2)  #bgr
                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

                if confidence < 80:    # 숫자가 작을 수록 명확
                    put_name = id
                else:
                    put_name = "None"


                confidence = "  {0}%".format(round(100 - confidence))

                cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, "Confidence"+str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 255), 1)

                tmp.append(id)

            cv2.imshow('camera', img)
            if cv2.waitKey(1) > 0: break

            # while문 내에서 db로 방문자 전송, 중복시 전송하지 않도록 코드 짜기

        visit_list.append(set(tmp))

        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()
        print(visit_list)

    return flask.render_template('done.html', d1=put_name, d2=now)


if __name__ == '__main__':
    # model = joblib.load('./trainer/tainer.pkl')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('./trainer/trainer.pkl')  # 학습된 모델을 불러옴
    # Flask 서비스 스타트
    app.run(host='0.0.0.0', port=5000, debug=True)

