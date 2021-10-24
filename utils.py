import numpy as np
import random
import os
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import pickle as pk
import seaborn as sns


def read_files(path, interpolation=True):
    f = open(path)
    s = ''
    for i, line in enumerate(f):
        s += line[:-1] + ' '
    rows_temp = s.split('}')[0].split('{')[-1].split(';')[:-1]
    cols_temp = s.split('}')[1].split('{')[-1].split(';')[:-1]
    rows = []
    cols = []
    for r in rows_temp:
        rows.append([eval(i) for i in r[r.index('[') + 1:-2].split(' ')])
    for c in cols_temp:
        cols.append([eval(i) for i in c[c.index('[') + 1:-2].split(' ')])
    if interpolation:
        rows_final = rows[0]
        cols_final = cols[0]

        total_diff = []
        for r, c in zip(rows, cols):
            r_diff = abs(np.diff(np.array(r)))[1:].tolist()
            c_diff = abs(np.diff(np.array(c)))[1:].tolist()
            all_dis = [(e ** 2 + c_diff[i] ** 2) ** 0.5 for i, e in enumerate(r_diff)]
            total_diff = total_diff + all_dis

        dis_gap = np.mean(total_diff)
        for i, e in enumerate(rows):
            if i == 0:
                continue
            rows_point_dis = rows[i][0] - rows[i - 1][-1]
            col_point_dis = cols[i][0] - cols[i - 1][-1]

            point_dis = (rows_point_dis ** 2 + col_point_dis ** 2) ** 0.5
            num_points = int(point_dis / dis_gap)

            if num_points >= 1:
                inter_row = [round(
                    rows[i - 1][-1] + (rows_point_dis / num_points) * ii + random.uniform(-1, 1) * 10 ** (
                                len(str(int(rows_point_dis / num_points))) - 2), 4) for ii, ee in
                             enumerate(range(num_points))]
                rows_final += inter_row[1:]
                inter_col = [round(cols[i - 1][-1] + (col_point_dis / num_points) * ii + random.uniform(-1, 1) * 10 ** (
                            len(str(int(col_point_dis / num_points))) - 2), 4) for ii, ee in
                             enumerate(range(num_points))]
                cols_final += inter_col[1:]
            rows_final += e
            cols_final += cols[i]
    else:
        rows_final = rows[0]
        for i in rows[1:]:
            rows_final += i
        cols_final = cols[0]
        for i in cols[1:]:
            cols_final += i

    return rows_final, cols_final


def generate_traj_feature_label(dir_path):
    label = []
    feature = []
    # dir_path = '/content/gdrive/MyDrive/FingerPen/Char74/Trj'
    g = os.walk(dir_path)
    for path,dir_list,path_list in g:
      for dir in dir_list:
        for sub_path,sub_dir_list,sub_file_list in os.walk(os.path.join(path,dir)):
          for f in sub_file_list:
            label.append(int(f[4:6]))
            x,y = read_files(os.path.join(sub_path,f))
            feature.append(np.array([x,y]).T.tolist())
            print(os.path.join(sub_path,f))
    np.save('/content/gdrive/MyDrive/FingerPen/Char74/feature.npy', feature)
    np.save('/content/gdrive/MyDrive/FingerPen/Char74/label.npy', label)


def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=plt.cm.Greens,normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(30, 30))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('./confusion_matrix.jpg')
    plt.show()


def plot_confuse(model, x_val, y_val, label):
    predictions = model.predict(x_val)
    predictions = predictions.argmax(axis=-1)
    truelabel = y_val.argmax(axis=-1)
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, normalize=False,target_names=label,title='Confusion Matrix')


def plot_pic(feature):
    x, y = zip(*feature)
    plt.scatter(x, y)
    plt.show()


def get_bigram_freq_data():

    data1 = pd.read_csv('/content/gdrive/MyDrive/FingerPen/googlebooks-eng-all-1gram-20090715-0.csv', sep='\t',
                        header=None)
    data1.columns = ['name', 'year', 'count', 'word_match', 'book_match']
    data1 = data1.groupby('name').agg('sum')

    data2 = pd.read_csv('/content/gdrive/MyDrive/FingerPen/googlebooks-eng-all-1gram-20090715-1.csv', sep='\t',
                        header=None)
    data2.columns = ['name', 'year', 'count', 'word_match', 'book_match']
    data2 = data2.groupby('name').agg('sum')

    data3 = pd.read_csv('/content/gdrive/MyDrive/FingerPen/googlebooks-eng-all-1gram-20090715-2.csv', sep='\t',
                        header=None)
    data3.columns = ['name', 'year', 'count', 'word_match', 'book_match']
    data3 = data3.groupby('name').agg('sum')

    data = pd.concat([data1, data2, data3])
    data.reset_index(inplace=True)

    bichargram_freq = {}
    labels_cate = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
    for i in labels_cate:
        bichargram_freq['#' + i] = 0
        for j in labels_cate:
            bichargram_freq[i + j] = 0

    def get_freq(row):
        word = str(row['name'])
        if not word.isalnum():
            return None
        num = row['count']
        for i, e in enumerate(word):
            if i == 0:
                try:
                    bichargram_freq['#' + e] += num
                except:
                    continue
            else:
                try:
                    bichargram_freq[word[i - 1:i + 1]] += num
                except:
                    continue

    data.apply(get_freq, axis=1)
    pk.dump(bichargram_freq, open('./bichargram_freq.pkl', 'wb'))


def get_trainsition_p(bichargram_freq,freq=True):
    labels_cate = [str(i) for i in range(10)] + [chr(i) for i in range(65,91)] + [chr(i) for i in range(97,123)]
    df_dict = {}
    for i in labels_cate:
        df_dict[i] = [bichargram_freq['#'+i]]
    for i in labels_cate:
        for j in labels_cate:
            df_dict[j].append(bichargram_freq[i+j])

    heap_df = pd.DataFrame(df_dict,index=['#']+labels_cate)
    if freq:
        x = heap_df.values
        sum_x = np.sum(x, axis=-1, keepdims=True)
        p_x = x / sum_x
        px_df = pd.DataFrame(p_x, index=['#'] + labels_cate, columns=labels_cate)
    else:
        px_df = heap_df

    return px_df


def get_heatmap(px_df):
    plt.figure(figsize=(30, 30))
    sns.heatmap(px_df, cmap=plt.cm.Greens)


def simulate_correction_p():
    labels_cate = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
    label2id = dict(zip(labels_cate, [i for i in range(len(labels_cate))]))

    smooth_t = np.random.uniform(0, 0.5, (6, 62))

    smooth_t[0][label2id['s']] += 0.6
    smooth_t[0][label2id['S']] += 0.4

    smooth_t[1][label2id['m']] += 0.8

    smooth_t[2][label2id['0']] += 0.6
    smooth_t[2][label2id['o']] += 0.4

    smooth_t[3][label2id['0']] += 0.6
    smooth_t[3][label2id['o']] += 0.4

    smooth_t[4][label2id['T']] += 0.6
    smooth_t[4][label2id['t']] += 0.4

    smooth_t[5][label2id['h']] += 0.8

    numerator = np.exp(smooth_t)
    denominator = np.sum(numerator, axis=1, keepdims=True)
    softmax_smooth_t = numerator / denominator
    return softmax_smooth_t


def get_test_data(test_path='./testData'):
    g = os.walk(test_path)
    result = []
    for path,dir_list,file_list in g:
        for f in file_list:
            if not 'txt' in f:
                continue
            print(f)
            sample = []
            f = open(os.path.join(path,f))
            for line in f.readlines():
                if '\n' not in line:
                    continue
                sample.append([eval(i) for i in line[1:-2].split(',')][::-1])
            result.append(sample)
    return result


def get_word_freq():
    data1 = pd.read_csv('../correction_data/googlebooks-eng-all-1gram-20090715-0.csv', sep='\t',
                        header=None)
    data1.columns = ['name', 'year', 'count', 'word_match', 'book_match']
    data1 = data1.groupby('name').agg('sum')

    data2 = pd.read_csv('../correction_data/googlebooks-eng-all-1gram-20090715-1.csv', sep='\t',
                        header=None)
    data2.columns = ['name', 'year', 'count', 'word_match', 'book_match']
    data2 = data2.groupby('name').agg('sum')

    data3 = pd.read_csv('../correction_data/googlebooks-eng-all-1gram-20090715-2.csv', sep='\t',
                        header=None)
    data3.columns = ['name', 'year', 'count', 'word_match', 'book_match']
    data3 = data3.groupby('name').agg('sum')

    data = pd.concat([data1, data2, data3])
    data.reset_index(inplace=True)
    pk.dump(data, open('./word_freq.pkl', 'wb'))
    return data[['name','count']]


def get_word_freq_P():
    word_freq = pk.load(open('./word_freq.pkl', 'rb'))
    word_freq_P = word_freq['count'].sum()
    word_freq['probability'] = word_freq['count'] / word_freq_P
    word = word_freq['name'].tolist()
    word_probability = word_freq['probability'].tolist()
    word2p = dict(zip(word,word_probability))
    pk.dump(word2p, open('./word2p.pkl', 'wb'))

if __name__ == '__main__':
    # get_test_data()
    get_word_freq_P()