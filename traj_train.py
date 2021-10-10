from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,CSVLogger
from tensorflow import keras
from traj_model import *
from trajectoryDataset import trajectoryData
from utils import *

if __name__ == '__main__':

    dataSet = trajectoryData(digit=False)
    feature, label = dataSet.char
    train_set_fea, train_set_label, val_set_fea, val_set_label, test_set_fea, test_set_label = dataSet.split_data(feature, label)

    # model = simpleLSTM(62, input_shape=train_set_fea.shape[1:])
    model_path = './traj_model_checkpoints'
    # model.fit(train_set_fea, train_set_label, validation_data=(val_set_fea, val_set_label), batch_size=128, epochs=50,
    #           callbacks=[EarlyStopping(verbose=True, patience=5, monitor='val_acc'),
    #                      ModelCheckpoint(model_path, monitor='val_acc', verbose=True, save_best_only=True),
    #                      CSVLogger(model_path + '.csv')])
    # model.save_weights(os.path.join(model_path,'model.h5'))

    model = simpleLSTM(62, input_shape=train_set_fea.shape[1:])
    model.load_weights(os.path.join(model_path,'model.h5'))

    labels_cate = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
    print(len(labels_cate))
    plot_confuse(model, val_set_fea, val_set_label, labels_cate)