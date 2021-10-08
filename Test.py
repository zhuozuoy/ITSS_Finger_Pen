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

        # variable
        self.gesture_str = 'None'       # Gesture recognition result
        self.rectangle_on = 0           # control rectangle show up
        self.rectangle_off = 0          # control rectangle show off
        self.clear = 0                  # clear all writing
        self.center = []                # handwriting center
        self.eraser = []                # eraser center
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

    def rectangle_logic(self,frame):
        threshold = 25
        if self.gesture_str == 'Four':
            self.rectangle_on += 1

        if self.rectangle_on > threshold:
            self.rectangle_show(frame)

        if self.gesture_str == 'Five':
            self.rectangle_off += 1

        if self.rectangle_off > threshold:
            self.rectangle_on = 0
            self.rectangle_off = 0

    def rectangle_show(self,frame):
        list = [[40, 150, 200, 500, (0, 255, 255)],
                [290, 150, 200, 500, (0, 255, 255)],
                [540, 150, 200, 500, (0, 255, 255)],
                [790, 150, 200, 500, (0, 255, 255)],
                [1040, 150, 200, 500, (0, 255, 255)]]
        for x, y, w, h, color in list:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    def writing_show(self,frame):
        for x,y in self.center:
            if (150<y<650) and ((40<x<240) or (290<x<490) or (540<x<740) or (790<x<990) or (1040<x<1240)):
                color = (255,0,0)
            else:
                color = (255,255,0)
            cv2.circle(frame, (x, y), 10, color, -1)

    def get_frame(self):
        success, image = self.video.read()

        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        results = self.hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.

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

                    #轨迹绘制
                    # print(hand_local[8][0])
                    center_x = math.floor(hand_local[8][0])
                    center_y = math.floor(hand_local[8][1])

                    eraser_x = math.floor((hand_local[8][0]+hand_local[12][0])/2)
                    eraser_y = math.floor((hand_local[8][1]+hand_local[12][1])/2)


                    angle_list = self.hand_angle(hand_local)
                    self.gesture_str = self.h_gesture(angle_list)
                    cv2.putText(frame,self.gesture_str,(100,100),0,3,(0,0,255),3)

                    if self.gesture_str == 'One':
                        self.clear = 0
                        self.center.append((center_x,center_y))

                    if self.gesture_str == 'Two':
                        self.eraser.append((eraser_x,eraser_y))
                        # cv2.circle(frame, (eraser_x, eraser_y), 10, (0, 0, 255), -1)
                        length = 20
                        cv2.rectangle(frame, (eraser_x-length, eraser_y-length), (eraser_x+length, eraser_y+length), (0,0,255), -1)

                        for i in range(eraser_x-length,eraser_x+length):
                            for j in range(eraser_y-length,eraser_y+length):
                                if (i,j) in self.center:
                                    self.center.remove((i,j))

                    # ret_list = list(set(self.center) ^ set(self.eraser))
                    # # ret_list = [item for item in self.center if item not in self.eraser]
                    # self.center = ret_list



        # show handwriting
        self.writing_show(frame)

        # show 5 rectangle
        self.rectangle_logic(frame)

        # Clear all handwriting
        if self.gesture_str == 'Three':
            self.clear += 1
        if self.clear > 80:
            memory = self.center
            self.center = []
            self.eraser = []
            self.clear = 0


        ret, jpeg = cv2.imencode('.jpg', frame)

        return jpeg.tobytes(), self.gesture_str
