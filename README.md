# ITSS_Finger_Pen
用Flask把frame放到网页上

安装环境  flask，mediapipe，opencv-python 后

运行app.py


Gesture Category: <br>
    "One"   -- Draw / Write <br>
    'Twp"   -- Eraser <br>
    "Three" -- Save the Result and Clear <br>
    "Four"  -- Show Writing Region( 5 Rectangles ) <br>
    "Five"  -- Hide Writing Region <br>
            <br>
Notes: <br>
    Make sure your whole hand are in the camera to get better gesture recognition result. <br>
    Only the words write in the region (Deep Blue Color) will be recognized. <br>
    Hold your gesture "Three" for a while to make sure you really want to clear your writing. <br>
