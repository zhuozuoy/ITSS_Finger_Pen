import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, render_template, Response, make_response
from Test import Gesture

app = Flask(__name__)


# 相机推流
def gen(Test):
    while True:
        frame, gesture = Test.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



# 相机喂流
@app.route('/video_feed')
def video_feed():
    return Response(gen(Gesture()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# 当前实时相机画面
@app.route('/')
def cur_camera():
    return render_template('cur_camer.html')


if __name__ == '__main__':
    app.run(host='localhost', debug=False)
