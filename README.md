# ITSS_Finger_Pen
用Flask把frame放到网页上

安装环境  flask，mediapipe，opencv-python 后

运行app.py


Gesture Category: 
    "One"   -- Draw / Write 
    'Twp"   -- Eraser 
    "Three" -- Save the Result and Clear
    "Four"  -- Show Writing Region( 5 Rectangles )
    "Five"  -- Hide Writing Region

Notes:
    Make sure your whole hand are in the camera to get better gesture recognition result.
    Only the words write in the region (Deep Blue Color) will be recognized.
    Hold your gesture "Three" for a while to make sure you really want to clear your writing.
