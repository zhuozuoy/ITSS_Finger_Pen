import cv2
import mediapipe as mp
import math


class Gesture(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75)

        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920);
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080);

    def __del__(self):
        self.video.release()

    def vector_2d_angle(self, v1, v2):
        '''
            求解二维向量的角度
        '''
        v1_x = v1[0]
        v1_y = v1[1]
        v2_x = v2[0]
        v2_y = v2[1]
        try:
            angle_ = math.degrees(math.acos(
                (v1_x * v2_x + v1_y * v2_y) / (((v1_x ** 2 + v1_y ** 2) ** 0.5) * ((v2_x ** 2 + v2_y ** 2) ** 0.5))))
        except:
            angle_ = 65535.
        if angle_ > 180.:
            angle_ = 65535.
        return angle_

    def hand_angle(self, hand_):
        '''
            获取对应手相关向量的二维角度,根据角度确定手势
        '''
        angle_list = []
        # ---------------------------- thumb 大拇指角度
        angle_ = self.vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[2][0])), (int(hand_[0][1]) - int(hand_[2][1]))),
            ((int(hand_[3][0]) - int(hand_[4][0])), (int(hand_[3][1]) - int(hand_[4][1])))
        )
        angle_list.append(angle_)
        # ---------------------------- index 食指角度
        angle_ = self.vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[6][0])), (int(hand_[0][1]) - int(hand_[6][1]))),
            ((int(hand_[7][0]) - int(hand_[8][0])), (int(hand_[7][1]) - int(hand_[8][1])))
        )
        angle_list.append(angle_)
        # ---------------------------- middle 中指角度
        angle_ = self.vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[10][0])), (int(hand_[0][1]) - int(hand_[10][1]))),
            ((int(hand_[11][0]) - int(hand_[12][0])), (int(hand_[11][1]) - int(hand_[12][1])))
        )
        angle_list.append(angle_)
        # ---------------------------- ring 无名指角度
        angle_ = self.vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[14][0])), (int(hand_[0][1]) - int(hand_[14][1]))),
            ((int(hand_[15][0]) - int(hand_[16][0])), (int(hand_[15][1]) - int(hand_[16][1])))
        )
        angle_list.append(angle_)
        # ---------------------------- pink 小拇指角度
        angle_ = self.vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[18][0])), (int(hand_[0][1]) - int(hand_[18][1]))),
            ((int(hand_[19][0]) - int(hand_[20][0])), (int(hand_[19][1]) - int(hand_[20][1])))
        )
        angle_list.append(angle_)
        return angle_list

    def h_gesture(self, angle_list):
        '''
            # 二维约束的方法定义手势
            # fist five gun love one six three thumbup yeah
        '''
        thr_angle = 65.
        thr_angle_thumb = 53.
        thr_angle_s = 49.
        gesture_str = None
        if 65535. not in angle_list:
            if (angle_list[0] > thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) and (
                    angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
                gesture_str = "One"
            elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (
                    angle_list[2] < thr_angle_s) and (angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
                gesture_str = "Two"
            elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (
                    angle_list[2] < thr_angle_s) and (angle_list[3] < thr_angle_s) and (angle_list[4] > thr_angle):
                gesture_str = "Three"
            elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] > thr_angle_s) and (
                    angle_list[2] < thr_angle_s) and (angle_list[3] < thr_angle_s) and (angle_list[4] < thr_angle):
                gesture_str = "Three"
            elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (
                    angle_list[2] < thr_angle_s) and (angle_list[3] < thr_angle_s) and (angle_list[4] < thr_angle):
                gesture_str = "Four"
            elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
                    angle_list[3] < thr_angle_s) and (angle_list[4] < thr_angle_s):
                gesture_str = "Five"
            elif (angle_list[0] < thr_angle_s) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (
                    angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
                gesture_str = "Thumb up"
            elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) and (
                    angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
                gesture_str = "L"
        return gesture_str

    def get_frame(self):
        success, image = self.video.read()

        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        results = self.hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        gesture_str = 'Nothing detected'

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                hand_local = []
                for i in range(21):
                    x = hand_landmarks.landmark[i].x * frame.shape[1]
                    y = hand_landmarks.landmark[i].y * frame.shape[0]
                    hand_local.append((x, y))
                if hand_local:
                    # Index fingertip
                    print(hand_local[8])
                    angle_list = self.hand_angle(hand_local)
                    gesture_str = self.h_gesture(angle_list)
                    cv2.putText(frame,gesture_str,(100,100),0,3,(0,0,255),3)
                    # cv2.putText(frame, gesture_str)

        list = [[40,150,200,500,(0,255,255)],
                [290,150,200,500,(0,255,255)],
                [540,150,200,500,(0,255,255)],
                [790,150,200,500,(0,255,255)],
                [1040,150,200,500,(0,255,255)]]
        for x,y,w,h,color in list:
            cv2.rectangle(frame,(x, y),(x + w, y + h),color,2)

        ret, jpeg = cv2.imencode('.jpg', frame)

        return jpeg.tobytes(), gesture_str
