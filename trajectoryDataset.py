import numpy as np
import pickle


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
        len_95 = int(np.percentile(np.array(feature_len), [95])[0])

        feature_processed = [self.preprocess(i, padding_length=len_95, standardization=False) for i in feature]

        label_num = len(set(label))
        labels_onehot = np.zeros((len(label), label_num))
        for i, e in enumerate(label):
            labels_onehot[i][e - 1] = 1

        return (feature_processed, labels_onehot)


    def preprocess(self, feature, originize=True, standardization=True, padding=True, padding_length=150):
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


    def data_augmentation(self, feature, label, multiple=3):
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
        return aug_feature, aug_label
