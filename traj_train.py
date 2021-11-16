from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,CSVLogger
from tensorflow import keras
from traj_model import *
from trajectoryDataset import trajectoryData
from utils import *
from correction_func import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

def model_evalute(prediction,y_test):
    print('Accuracy:', accuracy_score(y_test, prediction))
    print('F1 score:', f1_score(y_test, prediction, average='macro'))
    print('Recall:', recall_score(y_test, prediction, average='macro'))
    print('Precision:', precision_score(y_test, prediction, average='macro'))

if __name__ == '__main__':

    dataSet = trajectoryData(digit=False)
    # feature, label = dataSet.char
    # train_set_fea, train_set_label, val_set_fea, val_set_label, test_set_fea, test_set_label = dataSet.split_data(feature, label)
    # aug_feature, aug_label = dataSet.data_augmentation(train_set_fea.tolist(), train_set_label.tolist())
    # aug_feature, aug_label = dataSet.data_augmentation(aug_feature, aug_label, mode='start_point')

    # train_set_fea = np.array([dataSet.moving_ave(i, window_size=5) for i in train_set_fea.tolist()])
    # train_set_fea_diff = np.diff(train_set_fea, axis=1)
    #
    # val_set_fea = np.array([dataSet.moving_ave(i, window_size=5) for i in val_set_fea.tolist()])
    # val_set_fea_diff = np.diff(val_set_fea, axis=1)
    #
    # test_set_fea = np.array([dataSet.moving_ave(i, window_size=5) for i in test_set_fea.tolist()])
    # test_set_fea_diff = np.diff(test_set_fea, axis=1)

    # model = simpleLSTM(62, input_shape=train_set_fea.shape[1:])
    model_path = './traj_model_checkpoints'
    # model.fit(train_set_fea, train_set_label, validation_data=(val_set_fea, val_set_label), batch_size=128, epochs=50,
    #           callbacks=[EarlyStopping(verbose=True, patience=5, monitor='val_acc'),
    #                      ModelCheckpoint(model_path, monitor='val_acc', verbose=True, save_best_only=True),
    #                      CSVLogger(model_path + '.csv')])
    # model.save_weights(os.path.join(model_path,'model.h5'))

    feature = get_test_data()
    feature = [dataSet.preprocess(i,standardization=True) for i in feature]
    # plot_pic(feature[3])

    feature = np.array([dataSet.moving_ave(i, window_size=5) for i in feature])
    # plot_pic(feature[3])
    feature = np.array(feature)

    model = simpleLSTM(62, input_shape=feature.shape[1:])
    model.load_weights(os.path.join(model_path,'model_diff.h5'))

    dense_layer_model = Model(inputs=model.input,
                               outputs=model.get_layer('Dense_output').output)
    predictions = dense_layer_model.predict(feature)
    predictions = predictions + abs(np.min(predictions,axis=1,keepdims=True))
    print(predictions)
    predictions = predictions/np.sum(predictions,axis=1,keepdims=True)
    pred = predictions.argmax(axis=-1)

    labels_cate = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
    label2id = dict(zip(labels_cate,[i for i in range(len(labels_cate))]))
    id2label = dict(zip([i for i in range(len(labels_cate))],labels_cate))

    print([id2label[i] for i in pred])
    print([i[j] for i,j in zip(predictions,predictions.argmax(axis=-1))])

    true = [label2id[i] for i in ['c','l','e','a','r']]
    print(true)
    print([i[j] for i,j in zip(predictions,true)])

    result = correction_result(predictions,lambda_a=20)
    print(result)
    best_word = result[0][0]
    word_correction = word_correction(best_word,predictions)
    print(word_correction)
    # plot_confuse(model, val_set_fea, val_set_label, labels_cate)


