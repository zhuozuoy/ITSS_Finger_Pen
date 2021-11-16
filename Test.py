import cv2
import mediapipe as mp
import math
import numpy as np
import inspect, re
from trajectoryDataset import trajectoryData
from traj_model import simpleLSTM
from correction_func import *
from tensorflow.keras.models import Model
from image_based_recognition.img_predict import *

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
        self.video.set(cv2.CAP_PROP_FPS, 60)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

        # variable
        self.gesture_str = 'None'       # Gesture recognition result
        self.rectangle_on = 0           # control rectangle show up if > 40 show rectangle
        self.rectangle_off = 0          # control rectangle show off if > 80 hide rectangle
        self.clear = 0                  # clear all writing
        self.save = 0                   # save all writing
        self.center = []                # handwriting center
        self.eraser = []                # eraser center

        self.end = 0  # end writing
        self.rec = 0  # start recognition
        self.text = 0  # show recognition result

        self.mode = 0                   # mode choose. 0 writing 1 select from corrected options
        self.predict = ''
        self.fst_rst = ''
        self.scd_rst = ''
        self.final = ''

        # self.breakpoint = []
        # self.breakdot = (1,1)

        self.trajectory = []            # trajectory recognition
        self.trajectory1 = []
        self.trajectory2 = []
        self.trajectory3 = []
        self.trajectory4 = []
        self.trajectory5 = []

        self.trajDataset =  trajectoryData(digit=False)
        model_path = './traj_model_checkpoints'

        self.trajModel = simpleLSTM(62, input_shape=(self.trajDataset.len_95,2))
        self.trajModel.load_weights(os.path.join(model_path, 'model_diff.h5'))

        ckpt = torch.load(f'./image_based_recognition/ckpt/CNN_model_0.pkl', map_location=torch.device('cuda'))
        self.imageModel = SimpleCNN(62)
        self.imageModel.load_state_dict(ckpt)

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

    def gesture_logic(self):
        if self.gesture_str == 'One':
            return 'One -- Write'
        elif self.gesture_str == 'Two':
            return 'Two -- Eraser'
        elif self.gesture_str == 'Three':
            return 'Three -- Clear'
        elif self.gesture_str == 'Four':
            return 'Four -- Start Writing'
        elif self.gesture_str == 'Five':
            return 'Five -- End Writing'
        elif self.gesture_str == 'Thumb up':
            return 'Thumb up -- Save'

    def rectangle_logic(self,frame):
        if self.gesture_str == 'Four':
            self.rectangle_on += 1

        if self.rectangle_on > 40:
            self.end = 1
            self.rectangle_show(frame)
            self.rec = 1
            self.text = 0
            self.mode = 0
            self.final = ''


        if self.gesture_str == 'Five':
            self.rectangle_off += 1

        if self.rectangle_off > 80:
            self.rectangle_on = 0
            self.rectangle_off = 0
            self.end = 0

            if self.rec == 1:
                # recognition result
                if len(self.trajectory) == 0:
                    print('Do not have any trajectory to predict')
                    return
                self.predict, self.fst_rst, self.scd_rst = self.predict_correct(self.trajectory)
                self.text = 1
                self.rec = 0
                self.trajectory = []
                self.final = ''

        if self.text == 1:
            self.mode = 1
            self.recognition(frame)

    def rectangle_show(self,frame):
        # list = [[40, 150, 200, 500, (0, 255, 255)],
        #         [290, 150, 200, 500, (0, 255, 255)],
        #         [540, 150, 200, 500, (0, 255, 255)],
        #         [790, 150, 200, 500, (0, 255, 255)],
        #         [1040, 150, 200, 500, (0, 255, 255)]]
        # for x, y, w, h, color in list:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        y = 150
        w = 250
        h = 400
        color = (0,255,255)
        x = [40,340,640,940]
        for x in x:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    def writing_show(self,frame):
        for i in range(len(self.center)):
            if self.center[i]:
                last = self.center[i]
                break
        for x,y in self.center:
            # if (150<y<650) and ((40<x<240) or (290<x<490) or (540<x<740) or (790<x<990) or (1040<x<1240)):
            if (150 < y < 550) and ((40 < x < 290) or (340 < x < 590) or (640 < x < 890) or (940 < x < 1190)):
                color = (255,0,0)
            else:
                color = (255,255,0)

            # if (x,y) in self.breakpoint:
            #     color = (0,0,255)

            cv2.circle(frame, (x, y), 10, color, -1)

            # if last not in self.breakpoint:
            #     cv2.line(frame,(x,y),last,color,20)
            #     last = (x,y)
            # else:
            #     last = (x,y)

    def Trajectory_get(self):
        for x, y in self.center:
            y = y - 150
            if 0 < y < 400:
                # if 40 < x < 240:
                #     x = x - 40
                #     self.trajectory1.append((x, y))
                # elif 290 < x < 490:
                #     x = x - 290
                #     self.trajectory2.append((x, y))
                # elif 540 < x < 740:
                #     x = x - 540
                #     self.trajectory3.append((x, y))
                # elif 790 < x < 990:
                #     x = x - 790
                #     self.trajectory4.append((x, y))
                # elif 1040 < x < 1240:
                #     x = x - 1040
                #     self.trajectory5.append((x, y))
                if 40 < x < 290:
                    x = x - 40
                    self.trajectory1.append((x, y))
                elif 340 < x < 590:
                    x = x - 340
                    self.trajectory2.append((x, y))
                elif 640 < x < 890:
                    x = x - 640
                    self.trajectory3.append((x, y))
                elif 940 < x < 1190:
                    x = x - 940
                    self.trajectory4.append((x, y))

        # trajectory = [self.trajectory1, self.trajectory2, self.trajectory3, self.trajectory4, self.trajectory5]
        trajectory = [self.trajectory1, self.trajectory2, self.trajectory3, self.trajectory4]

        for traj in trajectory:
            if len(traj) != 0:
                self.trajectory.append(traj)

        self.trajectory1 = []
        self.trajectory2 = []
        self.trajectory3 = []
        self.trajectory4 = []
        # self.trajectory5 = []

        self.Trajectory_save(self.trajectory)

    def Trajectory_save(self,trajectory):
        # addr_list = ['trajectory1','trajectory2','trajectory3','trajectory4','trajectory5']
        # addr_list = ['trajectory1','trajectory2','trajectory3','trajectory4']
        # i = 0
        # for traj in trajectory:
        #     traj_addr = 'C:/Users/Ying/Downloads/' + addr_list[i] + '.txt'
        #     with open(traj_addr, mode='w') as f:
        #         for k, v in traj:
        #             dict_2_lst = []
        #             dict_2_lst.append("(" + str(k) + "," + str(v) + ")")
        #             f.write(','.join(dict_2_lst))
        #             f.write('\n')
        #
        #     # recognition = np.zeros((500, 200, 3), np.uint8)
        #     recognition = np.zeros((500, 250, 3), np.uint8)
        #     recognition.fill(255)
        #
        #     for x, y in traj:
        #         cv2.circle(recognition, (x, y), 10, (255, 0, 0), -1)
        #
        #     recg_addr = 'C:/Users/Ying/Downloads/' + addr_list[i] + '.jpg'
        #     cv2.imwrite(recg_addr, recognition)
        #
        #     i = i+1
        np.save("trajectory.npy", self.trajectory)

    def recognition(self,frame):
        predict = 'Recognition result:  ' + self.predict
        first = '1  ' + self.scd_rst[0][0]
        second = '2  ' + self.scd_rst[1][0]
        third = '3  ' + self.scd_rst[2][0]
        cv2.putText(frame, predict, (100, 200), 0, 2, (0, 255, 0), 3)
        if self.mode == 1:
            cv2.putText(frame, predict, (100,200), 0, 2, (0, 255, 0), 3)
            cv2.putText(frame, 'You may want to write:', (100, 300), 0, 2, (255, 255, 0), 3)
            cv2.putText(frame, first, (100, 400), 0, 2, (255, 255, 0), 3)
            cv2.putText(frame, second, (100, 500), 0, 2, (255, 255, 0), 3)
            cv2.putText(frame, third, (100, 600), 0, 2, (255, 255, 0), 3)

            if self.gesture_str == 'One':
                self.final = self.scd_rst[0][0]
            if self.gesture_str == 'Two':
                self.final = self.scd_rst[1][0]
            if self.gesture_str == 'Three':
                self.final = self.scd_rst[2][0]
            if self.gesture_str == 'Thumb up':
                self.final = self.predict

            final = 'Final result:  ' + self.final
            cv2.putText(frame, final, (100, 700), 0, 2, (0, 255, 0), 3)

    def correction(self, frame, predict):
        pass

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

                    gesture_text = self.gesture_logic()

                    if self.end != 0:
                        cv2.putText(frame,gesture_text,(100,100),0,2,(0,0,255),3)

                        if (self.mode == 0) and self.gesture_str == 'One':
                            self.clear = 0
                            self.save = 0
                            self.center.append((center_x,center_y))
                        #     self.breakdot = (center_x,center_y)
                        # else:
                        #     if self.breakdot not in self.breakpoint:
                        #         self.breakpoint.append(self.breakdot)



                        if self.mode == 0 and self.gesture_str == 'Two':
                            self.eraser.append((eraser_x,eraser_y))
                            length = 20
                            cv2.rectangle(frame, (eraser_x-length, eraser_y-length), (eraser_x+length, eraser_y+length), (0,0,255), -1)

                            for i in range(eraser_x-length,eraser_x+length):
                                for j in range(eraser_y-length,eraser_y+length):
                                    if (i,j) in self.center:
                                        self.center.remove((i,j))

                    else:
                        warning_text = 'Use Gesture 4 to start writing'
                        cv2.putText(frame, warning_text, (100, 100), 0, 2, (0, 0, 255), 3)


        # show handwriting
        self.writing_show(frame)

        # show 5 rectangle
        self.rectangle_logic(frame)

        # Clear all handwriting
        if self.mode == 0 and self.gesture_str == 'Three':
            self.clear += 1
        if self.clear > 80:
            memory = self.center

            self.center = []
            self.eraser = []
            self.clear = 0

        if self.mode ==0 and self.gesture_str == 'Thumb up':
            self.save += 1
        if self.save > 80:
            self.Trajectory_get()
            self.center = []
            self.eraser = []
            self.save = 0


        ret, jpeg = cv2.imencode('.jpg', frame)

        return jpeg.tobytes(), self.gesture_str


    def predict_correct(self,traj_list,target_str='air',lambda_a=1,lambda_b=1):

        traj_predictions = self.pred_trajatory(traj_list)
        image_predictions = self.pred_image(traj_list)
        image_predictions = np.reshape(image_predictions,(image_predictions.shape[0],image_predictions.shape[-1]))
        image_predictions = max_min_normalization(image_predictions)
        predictions = lambda_a*traj_predictions+lambda_b*image_predictions
        pred = predictions.argmax(axis=-1)

        labels_cate = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
        label2id = dict(zip(labels_cate, [i for i in range(len(labels_cate))]))
        id2label = dict(zip([i for i in range(len(labels_cate))], labels_cate))

        predict_result = ''.join([id2label[i] for i in pred])
        print(predict_result)
        print([i[j] for i, j in zip(predictions, predictions.argmax(axis=-1))])

        true = [label2id[i] for i in target_str]
        print(true)
        print([i[j] for i, j in zip(predictions, true)])

        first_corr_result = correction_result(predictions, lambda_a=20)
        print(first_corr_result)
        best_word = first_corr_result[0][0]
        second_corr_result = word_correction(best_word, predictions)
        print(second_corr_result)

        return predict_result,first_corr_result,second_corr_result


    def pred_trajatory(self,traj_list):
        traj_list = [[list(e)[::-1] for e in f] for f in traj_list]
        print(traj_list)
        feature = [self.trajDataset.preprocess(i, standardization=True) for i in traj_list]
        feature = np.array([self.trajDataset.moving_ave(i, window_size=5) for i in feature])
        feature = np.array(feature)

        dense_layer_model = Model(inputs=self.trajModel.input,outputs=self.trajModel.get_layer('Dense_output').output)

        predictions = dense_layer_model.predict(feature)
        predictions = predictions + abs(np.min(predictions, axis=1, keepdims=True))

        predictions = predictions / np.sum(predictions, axis=1, keepdims=True)
        return predictions


    def pred_image(self,trajList):
        with torch.no_grad():
            self.imageModel.eval()
            # load mapping txt
            # dict_tmp = load_mapping_file('./mappings/emnist-byclass-mapping.txt')

            # load the trajectory-based test sets and model file
            # traj_dir = "../testData"
            # trajList = []
            # for files in os.listdir(traj_dir):
            #     if os.path.splitext(files)[1] == '.txt':
            #         trajList.append(files)
            final_result = []
            # trajList = os.listdir(traj_dir)
            # trajList.sort(key=lambda x: int(x.replace("trajectory", "").split('.')[0]))
            for idx in range(0, len(trajList)):
                # print(os.path.join(traj_dir, trajList[idx]))
                x, y= zip(*trajList[idx])
                t = np.linspace(0.0, 1.0, 25)
                plt.figure(figsize=(1, 1))
                data = gen_bezuer(x, y, t)
                ax = plt.gca()
                ax.invert_yaxis()
                plt.axis('off')
                im = plt.plot(data[0], data[1], 'bo-')
                save_pth = os.path.join('./image_based_recognition/bezier_imgs/', str(idx+1) + '.jpg')
                plt.savefig(save_pth)
                plt.close()
                im = Image.open(save_pth).resize((28, 28))
                im = im.convert('L')

                im_data = np.array(im)
                im_data = 255 - im_data
                im_data = torch.from_numpy(im_data).float()
                im_data = im_data.view(1, 1, 28, 28)
                out = self.imageModel(im_data)
                # print(out)
                # print(type(out))
                _, pred = torch.max(out, 1)
                # 62-dimensional features
                features = out.detach().numpy()
                min = features.min()
                # print(min)
                features -= min
                final_result.append(features)
        return np.array(final_result)
                # print('predict: {}'.format(chr(int(dict_tmp[int(pred)]))))
    #2021-10-24 16:01
