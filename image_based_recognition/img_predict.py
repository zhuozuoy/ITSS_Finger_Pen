# -*- coding: utf-8 -*-
import os
import math
import torch
from img_model import SimpleCNN
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

NUM_CLASS = 62
TRAIN_EPOCH = 100
BATCH_SIZE = 1024
LEARNING_RATE = 1e-2


def gen_con_dot(a, b):
    A = np.array([a[0], b[0]])
    B = np.array([a[1], b[1]])
    C = np.array([a[2], b[2]])
    AB = B - A
    CB = B - C
    lab = math.hypot(AB[0], AB[1])
    lcb = math.hypot(CB[0], CB[1])
    e = (A + B) / 2
    f = (B + C) / 2
    D = f + (e - f) * lcb / (lcb + lab)
    DB = B - D
    E = e + DB
    F = f + DB
    return [[E[0].tolist(), F[0].tolist()], [E[1].tolist(), F[1].tolist()]]


def gen_who_dot(a, b):
    s = gen_con_dot([a[0], a[1], a[2]], [b[0], b[1], b[2]])
    x = [a[0], (s[0][0] + a[0]) / 2]
    y = [b[0], (s[1][0] + b[0]) / 2]
    for i in range(len(a) - 2):
        s = gen_con_dot([a[i], a[i + 1], a[i + 2]], [b[i], b[i + 1], b[i + 2]])
        x.extend([s[0][0], a[i + 1], s[0][1]])
        y.extend([s[1][0], b[i + 1], s[1][1]])
    x.extend([(a[len(a) - 1] + x[len(x) - 1]) / 2, a[len(a) - 1]])
    y.extend([(b[len(b) - 1] + y[len(y) - 1]) / 2, b[len(b) - 1]])
    return [x, y]


def gen_sep_bezier(a, b, t):
    x = a[0] * (1 - t) ** 3 + 3 * a[1] * t * (1 - t) ** 2 + 3 * a[2] * t * t * (1 - t) + a[3] * t ** 3
    y = b[0] * (1 - t) ** 3 + 3 * b[1] * t * (1 - t) ** 2 + 3 * b[2] * t * t * (1 - t) + b[3] * t ** 3
    return [x.tolist(), y.tolist()]


def gen_bezuer(a, b, t):
    m = gen_who_dot(a, b)[0]
    n = gen_who_dot(a, b)[1]

    p = [m[0], m[1], m[2], m[3]]
    q = [n[0], n[1], n[2], n[3]]
    x = gen_sep_bezier(p, q, t)[0]
    y = gen_sep_bezier(p, q, t)[1]

    for i in range(len(m) // 3 - 1):
        p = [m[3 * i + 3], m[3 * i + 4], m[3 * i + 5], m[3 * i + 6]]
        q = [n[3 * i + 3], n[3 * i + 4], n[3 * i + 5], n[3 * i + 6]]
        x.extend(gen_sep_bezier(p, q, t)[0])
        y.extend(gen_sep_bezier(p, q, t)[1])

    return [x, y]


def load_mapping_file(file_pth):
    dict_tmp = {}
    mapping_file = open(file_pth, 'r')

    for line in mapping_file.readlines():
        line = line.strip()
        k = int(line.split(' ')[0])
        v = int(line.split(' ')[1])
        dict_tmp[k] = v

    mapping_file.close()
    #print(dict_tmp)

    return dict_tmp


if __name__ == '__main__':

    # load the image-based test sets and model file
    ckpt = torch.load(f'./ckpt/CNN_model_0.pkl', map_location=torch.device('cpu'))
    model = SimpleCNN(NUM_CLASS)
    model.load_state_dict(ckpt)

    with torch.no_grad():
        model.eval()
        # load mapping txt
        dict_tmp = load_mapping_file('./mappings/emnist-byclass-mapping.txt')

        # load the trajectory-based test sets and model file
        traj_dir = "../testData"
        trajList = []
        for files in os.listdir(traj_dir):
            if os.path.splitext(files)[1] == '.txt':
                trajList.append(files)

        trajList = os.listdir(traj_dir)
        trajList.sort(key=lambda x: int(x.replace("trajectory","").split('.')[0]))
        for idx in range(0, len(trajList)):
            with open(os.path.join(traj_dir, trajList[idx])) as f:
                x, y = [], []
                for line in f.readlines():
                    line = line.strip()
                    x.append(line.split('(')[1].split(')')[0].split(',')[0])
                    y.append(line.split('(')[1].split(')')[0].split(',')[1])
                x = [float(a) for a in x]
                y = [float(b) for b in y]
                t = np.linspace(0.0, 1.0, 25)

            f.close()

            plt.figure(figsize=(1, 1))
            data = gen_bezuer(x, y, t)
            ax = plt.gca()
            ax.invert_yaxis()
            plt.axis('off')
            im = plt.plot(data[0], data[1], 'bo-')
            save_pth = os.path.join('./bezier_imgs/', trajList[idx].split('.')[0].split('y')[1] + '.jpg')
            plt.savefig(save_pth)
            plt.show()
            im = Image.open(save_pth).resize((28, 28))
            im = im.convert('L')

            im_data = np.array(im)
            im_data = 255 - im_data

            im_data = torch.from_numpy(im_data).float()
            im_data = im_data.view(1, 1, 28, 28)
            out = model(im_data)
            _, pred = torch.max(out, 1)

            # 62-dimensional features
            features = out.detach().numpy()
            min = features.min()
            print(min)
            features -= min
            print(features)

            print('predict: {}'.format(chr(int(dict_tmp[int(pred)]))))
