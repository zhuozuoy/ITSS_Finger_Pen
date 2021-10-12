import numpy as np
import pickle
from utils import plot_pic
import random

class trajectoryData(object):

    def __init__(self, char=True, digit=True):
        if char:
            self.char = self.load_char74()
        if digit:
            self.digits = self.load_digits()

    def load_digits(self):
        with open('/content/gdrive/MyDrive/RTD Dataset/features', 'rb') as fp:
            feature = pickle.load(fp)
        feature = np.resize(feature, new_shape=(feature.shape[0], int(feature.shape[1] / 2), 2))
        with open('/content/gdrive/MyDrive/RTD Dataset/labels', 'rb') as fp:
            labels = pickle.load(fp)
        return (feature, labels)

    def load_char74(self):
        feature = np.load('./traj_dataset/feature.npy', allow_pickle=True)
        feature = feature.tolist()
        label = np.load('./traj_dataset/label.npy', allow_pickle=True)
        label = label.tolist()

        feature_len = [len(i) for i in feature]
        self.len_95 = int(np.percentile(np.array(feature_len), [95])[0] * 1.5)

        feature_processed = [self.preprocess(i, standardization=True) for i in feature]

        # plot_pic(feature_processed[201])
        # plot_pic(self.moving_ave(feature_processed[201], window_size=5))

        label_num = len(set(label))
        labels_onehot = np.zeros((len(label), label_num))
        for i, e in enumerate(label):
            labels_onehot[i][e - 1] = 1

        return (feature_processed, labels_onehot)

    def preprocess(self, feature, originize=True, standardization=True, padding=True, padding_length=None):
        if padding_length == None:
            padding_length = self.len_95
        result = feature
        if originize:
            result = []
            origin_x, origin_y = feature[0]
            for i, j in feature:
                result.append([i - origin_x, j - origin_y])

        if standardization:
            data = np.array(result)
            mean = np.mean(data)
            sigma = np.std(data)
            data = (data - mean) / sigma
            result = data.tolist()

        if padding:
            if len(feature) < padding_length:
                for i in range(padding_length - len(feature)):
                    result.append([0, 0])
            elif len(feature) > padding_length:
                result = result[:padding_length]

        return result

    def split_data(self, feature, label, random_seed=2021, train_val_test_ratio=[0.7, 0.29, 0.01]):
        data_set = list(zip(feature, label))
        np.random.seed(random_seed)
        np.random.shuffle(data_set)
        sample_len = len(data_set)
        print('Total_train_samples:', sample_len)

        train_set = data_set[:int(train_val_test_ratio[0] * sample_len)]
        val_set = data_set[int(train_val_test_ratio[0] * sample_len):int(sum(train_val_test_ratio[:2]) * sample_len)]
        test_set = data_set[int(sum(train_val_test_ratio[:2]) * sample_len):]

        train_set_fea, train_set_label = zip(*train_set)
        val_set_fea, val_set_label = zip(*val_set)
        test_set_fea, test_set_label = zip(*test_set)

        train_set_fea = np.array(list(train_set_fea))
        train_set_label = np.array(list(train_set_label))
        val_set_fea = np.array(list(val_set_fea))
        val_set_label = np.array(list(val_set_label))
        test_set_fea = np.array(list(test_set_fea))
        test_set_label = np.array(list(test_set_label))

        print('valSet:feature shape{},label shape{}'.format(val_set_fea.shape, val_set_label.shape))
        print('trainSet:feature shape{},label shape{}'.format(train_set_fea.shape, train_set_label.shape))
        print('testSet:feature shape{},label shape{}'.format(test_set_fea.shape, test_set_label.shape))

        return train_set_fea, train_set_label, val_set_fea, val_set_label, test_set_fea, test_set_label

    def data_augmentation(self, feature, label, multiple=3, mode='rotate'):  # mode='rotate','noise','start_point'
        if mode == 'noise':
            aug_feature = []
            aug_label = []
            for i, f in enumerate(feature):
                aug_feature.append(f)
                aug_label.append(label[i])
                for m in range(multiple):
                    temp = np.array(f)
                    aug = (temp != 0) * np.random.rand(temp.shape[0], temp.shape[1]) * 10 ** -2 + temp
                    aug_feature.append(aug.tolist())
                    aug_label.append(label[i])

        elif mode == 'rotate':
            aug_feature = []
            aug_label = []
            for i, f in enumerate(feature):
                aug_feature.append(f)
                aug_label.append(label[i])
                for m in range(multiple):
                    aug = self.rotate_data(f, theta=random.randint(-45, 45))
                    aug_feature.append(aug)
                    aug_label.append(label[i])

        elif mode == 'start_point':
            aug_feature = []
            aug_label = []
            for i, f in enumerate(feature):
                aug_feature.append(f)
                aug_label.append(label[i])
                for m in range(multiple):
                    aug = self.add_random_startpoints(f)
                    aug_feature.append(aug)
                    aug_label.append(label[i])

        return aug_feature, aug_label

    def rotate_data(self, point_list, theta=None):
        point_matrix = np.array(point_list)
        rotate_matrix = np.array([[np.cos(theta * np.pi / 180), -np.sin(theta * np.pi / 180)],
                                  [np.sin(theta * np.pi / 180), np.cos(theta * np.pi / 180)]])
        point_matrix = np.dot(point_matrix, rotate_matrix)
        return point_matrix.tolist()

    def add_random_startpoints(self, point_list):

        rows, cols = zip(*point_list)
        rows_final = list(rows)
        cols_final = list(cols)

        left_point = min(rows)
        right_point = max(rows)
        up_point = max(cols)
        down_point = min(cols)

        rsp_x = random.uniform(left_point, right_point)
        rsp_y = random.uniform(down_point - (up_point - down_point), down_point)

        rows_point_dis = rows[0] - rsp_x
        col_point_dis = cols[0] - rsp_y

        r_diff = abs(np.diff(np.array(rows)))[1:].tolist()
        c_diff = abs(np.diff(np.array(cols)))[1:].tolist()
        all_dis = [(e ** 2 + c_diff[i] ** 2) ** 0.5 for i, e in enumerate(r_diff)]
        dis_gap = np.mean(all_dis)

        point_dis = (rows_point_dis ** 2 + col_point_dis ** 2) ** 0.5
        num_points = int(point_dis / dis_gap)

        if num_points >= 1:
            inter_row = [round(rsp_x + (rows_point_dis / num_points) * ii + random.uniform(-5, 5) * 10 ** (
                        len(str(int(rows_point_dis / num_points))) - 3), 4)
                         for ii, ee in enumerate(range(num_points))]
            rows_final = inter_row[1:] + rows_final
            inter_col = [round(rsp_y + (col_point_dis / num_points) * ii + random.uniform(-5, 5) * 10 ** (
                    len(str(int(col_point_dis / num_points))) - 3), 4) for ii, ee in
                         enumerate(range(num_points))]
            cols_final = inter_col[1:] + cols_final

            zero_len = sum(i == 0 for i in rows)
            add_len = len(inter_row) - 1
            if zero_len >= add_len:
                return list(zip(rows_final, cols_final))[:self.len_95]
            else:
                # print(len(list(zip(rows_final,cols_final))[add_len-zero_len:]))
                return list(zip(rows_final, cols_final))[add_len - zero_len:add_len - zero_len + self.len_95]
        else:
            return point_list

    def moving_ave(self, data, window_size):
        x, y = zip(*data)
        x = np.array(x)
        y = np.array(y)
        window = np.ones(int(window_size)) / float(window_size)
        x_result = np.convolve(x, window, 'same').tolist()
        y_result = np.convolve(y, window, 'same').tolist()
        return list(zip(x_result, y_result))

