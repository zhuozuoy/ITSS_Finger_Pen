# ITSS_Finger_Pen

[![Finger_Pen video](https://github.com/zhuozuoy/ITSS_Finger_Pen/blob/main/pic/Snapshot_1.png)](https://www.youtube.com/watch?v=DFGwUp_naoQ)

## Abstract
Signing and writing are inevitable events during social communication. However, this common behavior becomes high risky in public because of the pandemic. To solve this problem and decrease the spread of COVID-19, we have an idea to create a vision writing system. In this paper, we use MediaPipe hand detection API to detect people's hands in the camera. Based on finger landmarks, we can recognize users' gestures using vector included angles and achieve different control effects like writing, erasing, choosing and etc. After getting users' hand-writing, we combine trajectory-based and image-based methods to make recognition. For trajectory-based method, a double-layers stacked LSTM model trained by Char74k dataset is used for predicting. For image-based method, received trajectories from writing module are transformed to images using Bézier curve and a CNN model trained by EMNIST dataset is used for predicting. After recognition, we also use modified HMM and Bayesian model to correct recognition result based on word frequency. During all contact-less operating processes, users only need to use hand and gesture to determine when to write, erase, save and recognize based on our operating system and choose desired result within both recognition result and correction result.
## User Guide
1. pip install -r requirements.txt
2. python app.py

Gesture Category: <br>
    "One"   -- Draw / Write <br>
    'Two"   -- Eraser <br>
    "Three" -- Save the Result and Clear <br>
    "Four"  -- Show Writing Region( 4 Rectangles ) <br>
    "Five"  -- Hide Writing Region <br>
            <br>
Notes: <br>
    Make sure your whole hand are in the camera to get better gesture recognition result. <br>
    Only the words write in the region (Deep Blue Color) will be recognized. <br>
    Hold your gesture "Three" for a while to make sure you really want to save your writing and clear screen. <br>
## PROJECT REPORT / PAPER
Refer to project report at Github Folder: [ProjectReport](https://github.com/zhuozuoy/ITSS_Finger_Pen/report/Graduate_Certificate_Intelligent_Sensing_Systems_Practice_Module_Report_Group7.pdf)

