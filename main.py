# https://wings2pc.tistory.com/entry/%EC%9B%B9-%EC%95%B1%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%ED%94%8C%EB%9D%BC%EC%8A%A4%ED%81%ACPython-Flask?category=777829
# https://go-guma.tistory.com/9
import flask
from flask import Flask, request, render_template
import cv2
from datetime import datetime


app = Flask(__name__)
# 메인페이지 - url 요청시 기본 index.html로 이동 (렌더링)
# @app.route("/", methods=['POST', 'GET'])
@app.route("/", methods=['POST','GET'])
def cam_main():
    return render_template('camera.html')

@app.route("/camera", methods=['POST','GET'])
def cam_main2():
    return render_template('camera.html')


@app.route("/update", methods=['POST', 'GET'])    # 변경가능, 임시
def makemodel():
    return flask.render_template('camera.html', d0="업데이트를 완료했습니다!")

# 데이터
# 데이터 예측 처리
visit_list=[]
tmp = []
@app.route('/predict', methods=['POST', 'GET'])

########################## 카메라 켜서 예측하는 부분 #################################
def make_prediction():
    if request.method == 'POST':

        # 업로드 파일 처리 분기  # 카메라로 이 부분을 받아오면 될거 같음

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('./trainer/trainer.pkl')           # 학습된 모델을 불러옴
        cascadePath = 'haarcascade_frontalface_default.xml'
        faceCascade = cv2.CascadeClassifier(
            r"C:\Users\Yewon\anaconda3\envs\py38\Library\etc\haarcascades\haarcascade_frontalface_default.xml")

        font = cv2.FONT_HERSHEY_SIMPLEX

        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        minW = 0.1 * cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        minH = 0.1 * cam.get(cv2.CAP_PROP_FRAME_HEIGHT)


        while True:

            # DB내에서 넘어온 사용자 리스트에 있는지 확인할때까지만 loop 돌리도록 코드 수정 필요
            # 이후 조건문

            # start -= 1
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

                if confidence < 60:
                    put_name = id
                    # name을 거치지 않고 id를 출력하게 함. 이후 출력된 id는 db로 보내야함
                else:
                    put_name = "확인되지 않은 방문자"


                confidence = "  {0}%".format(round(100 - confidence))

                # cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
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
    # Flask 서비스 스타트
    app.run(host='0.0.0.0', port=5000, debug=True)

