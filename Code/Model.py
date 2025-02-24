"""
@author: Jiaxin Ye
@contact: jiaxin-ye@foxmail.com
"""
import numpy as np
import tensorflow.keras.backend as K
import os
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Layer, Dense, Input
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
from DeepSVDD import autoFilter
from Common_Model import Common_Model
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import datetime
import pandas as pd
import copy
import pickle

from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mode

from TIMNET import TIMNET


def smooth_labels(labels, factor=0.1):
    # smooth the labels
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels


class WeightLayer(Layer):
    def __init__(self, **kwargs):
        super(WeightLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], 1),
                                      initializer='uniform',
                                      trainable=True)
        super(WeightLayer, self).build(input_shape)

    def call(self, x):
        tempx = tf.transpose(x, [0, 2, 1])
        x = K.dot(tempx, self.kernel)
        x = tf.squeeze(x, axis=-1)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


class TIMNET_Model(Common_Model):
    def __init__(self, args, input_shape, class_label, **params):
        super(TIMNET_Model, self).__init__(**params)
        self.args = args
        self.data_shape = input_shape
        self.num_classes = len(class_label)
        self.class_label = class_label
        self.matrix = []
        self.eva_matrix = []
        self.acc = 0
        print("TIMNET MODEL SHAPE:", input_shape)

    def create_model(self):
        self.inputs = Input(shape=(self.data_shape[0], self.data_shape[1]))
        self.multi_decision = TIMNET(nb_filters=self.args.filter_size,
                                     kernel_size=self.args.kernel_size,
                                     nb_stacks=self.args.stack_size,
                                     dilations=self.args.dilation_size,
                                     dropout_rate=self.args.dropout,
                                     activation=self.args.activation,
                                     return_sequences=True,
                                     name='TIMNET')(self.inputs)

        self.decision = WeightLayer()(self.multi_decision)
        self.predictions = Dense(self.num_classes, activation='softmax')(self.decision)
        self.model = Model(inputs=self.inputs, outputs=self.predictions)

        self.model.compile(loss="categorical_crossentropy",
                           optimizer=Adam(learning_rate=self.args.lr, beta_1=self.args.beta1, beta_2=self.args.beta2,
                                          epsilon=1e-8),
                           metrics=['accuracy'])
        print("Temporal create succes!")

    def train(self, x, y, filtering=True):
        filepath = self.args.model_path
        resultpath = self.args.result_path

        if not os.path.exists(filepath):
            os.mkdir(filepath)
        if not os.path.exists(resultpath):
            os.mkdir(resultpath)

        now = datetime.datetime.now()
        now_time = datetime.datetime.strftime(now, '%Y-%m-%d_%H-%M-%S')
        kfold = KFold(n_splits=self.args.split_fold, shuffle=True, random_state=self.args.random_seed)
        avg_accuracy_inlier = 0
        avg_loss_inlier = 0
        avg_accuracy_outlier = 0
        avg_loss_outlier = 0
        avg_accuracy = 0
        avg_loss = 0
        for index, (train, test) in enumerate(kfold.split(x, y)):
            train, test_inlier, test_score = autoFilter.filter_data(filtering, train, test, x, y, (index+1), threshold=0.5)

            self.create_model()
            y_train = smooth_labels(copy.deepcopy(y[train]), 0.1)
            folder_address = filepath + self.args.data + "_" + str(self.args.random_seed) + "_" + now_time
            if not os.path.exists(folder_address):
                os.mkdir(folder_address)
            weight_path = folder_address + '/' + str(self.args.split_fold) + "-fold_weights_best_" + str(
                (index+1)) + ".hdf5"
            checkpoint = callbacks.ModelCheckpoint(weight_path, verbose=1, save_weights_only=True, save_best_only=False)
            max_acc = 0
            best_eva_list_inlier = []
            best_eva_list_outlier = []
            h = self.model.fit(x[train], y_train, validation_data=(x[test], y[test]), batch_size=self.args.batch_size,
                               epochs=self.args.epoch, verbose=1, callbacks=[checkpoint])
            self.model.load_weights(weight_path)

            print(set(test), set(test_inlier))
            test_outlier = list(set(test) - set(test_inlier))

            # evaluate test inlier
            best_eva_list_inlier = self.model.evaluate(x[test_inlier], y[test_inlier])
            avg_loss_inlier += best_eva_list_inlier[0]
            avg_accuracy_inlier += best_eva_list_inlier[1]
            print("At fold: " + str((index+1)) + '. Inlier Model evaluation: ', best_eva_list_inlier, "   Now ACC:",
                  str(round(avg_accuracy_inlier * 10000) / 100 / (index+1)))

            # evaluate outlier
            if len(test_outlier) > 0:
                best_eva_list_outlier = self.model.evaluate(x[test_outlier], y[test_outlier])
                avg_loss_outlier += best_eva_list_outlier[0]
                avg_accuracy_outlier += best_eva_list_outlier[1]
                print("At fold: " + str((index+1)) + '. Outlier Model evaluation: ', best_eva_list_outlier, "   Now ACC:",
                      str(round(avg_accuracy_outlier * 10000) / 100 / (index+1)))
            else:
                print("Skipped outlier evaluation: empty list!")

            # evaluate all
            best_eva_list = self.model.evaluate(x[test], y[test])
            avg_loss += best_eva_list[0]
            avg_accuracy += best_eva_list[1]
            print("At fold: " + str((index+1)) + '_Model evaluation: ', best_eva_list, "   Now ACC:",
                  str(round(avg_accuracy * 10000) / 100 / (index+1)))

            # Train KNN model on the training set
            knn = KNeighborsClassifier(n_neighbors=5)  # you can adjust the number of neighbors if needed
            # print(x[train].shape, np.argmax(y[train], axis=1).shape)
            print(test_outlier)
            x_train_reshaped = x[test_outlier].reshape(x[test_outlier].shape[0], -1)
            knn.fit(x_train_reshaped, np.argmax(y[test_outlier], axis=1))

            # Predict using the main model
            y_pred_best = self.model.predict(x[test])
            y_pred_labels_main = np.argmax(y_pred_best, axis=1)

            # Predict using KNN
            y_pred_labels_knn = knn.predict(x[test].reshape(x[test].shape[0], -1))

            x_test_score = test_score
            print(f"Outlierness score: {x_test_score}")

            y_pred_labels = [y_pred_labels_main[j] if score < 0.5 else y_pred_labels_knn[j]
                             for j, score in enumerate(x_test_score)]

            y_true_labels = np.argmax(y[test], axis=1)

            # Extract indices of misclassified and correctly classified samples
            misclassified_indices = np.where(y_pred_labels != y_true_labels)[0]
            correctly_classified_indices = np.where(y_pred_labels == y_true_labels)[0]
            # print(misclassified_indices, correctly_classified_indices)

            # Extract test_score values for both sets of samples
            # print(test_score)
            misclassified_scores = [test_score[i] for i in misclassified_indices]
            correctly_classified_scores = [test_score[i] for i in correctly_classified_indices]

            # Calculate average scores
            avg_misclassified_score = np.mean(misclassified_scores)
            avg_correctly_classified_score = np.mean(correctly_classified_scores)

            print(f"Average test_score for misclassified samples: {avg_misclassified_score}")
            print(f"Average test_score for correctly classified samples: {avg_correctly_classified_score}")

            self.matrix.append(confusion_matrix(np.argmax(y[test], axis=1), np.argmax(y_pred_best, axis=1)))
            em = classification_report(np.argmax(y[test], axis=1), np.argmax(y_pred_best, axis=1),
                                       target_names=self.class_label, output_dict=True)
            self.eva_matrix.append(em)
            print(classification_report(np.argmax(y[test], axis=1), np.argmax(y_pred_best, axis=1),
                                        target_names=self.class_label))

        print("Average ACC:", avg_accuracy / self.args.split_fold)
        self.acc = avg_accuracy / self.args.split_fold
        writer = pd.ExcelWriter(resultpath + self.args.data + '_' + str(self.args.split_fold) + 'fold_' + str(
            round(self.acc * 10000) / 100) + "_" + str(self.args.random_seed) + "_" + now_time + '.xlsx')
        for i, item in enumerate(self.matrix):
            temp = {}
            temp[" "] = self.class_label
            for j, l in enumerate(item):
                temp[self.class_label[j]] = item[j]
            data1 = pd.DataFrame(temp)
            data1.to_excel(writer, sheet_name=str(i), encoding='utf8')

            df = pd.DataFrame(self.eva_matrix[i]).transpose()
            df.to_excel(writer, sheet_name=str(i) + "_evaluate", encoding='utf8')
        writer.save()
        writer.close()

        K.clear_session()
        self.matrix = []
        self.eva_matrix = []
        self.acc = 0
        self.trained = True

    def test(self, x, y, path):
        i = 1
        kfold = KFold(n_splits=self.args.split_fold, shuffle=True, random_state=self.args.random_seed)
        avg_accuracy = 0
        avg_loss = 0
        x_feats = []
        y_labels = []
        for train, test in kfold.split(x, y):
            self.create_model()
            weight_path = path + '/' + str(self.args.split_fold) + "-fold_weights_best_" + str(i) + ".hdf5"
            self.model.fit(x[train], y[train], validation_data=(x[test], y[test]), batch_size=64, epochs=0, verbose=0)
            self.model.load_weights(weight_path)  # +source_name+'_single_best.hdf5')
            best_eva_list = self.model.evaluate(x[test], y[test])
            avg_loss += best_eva_list[0]
            avg_accuracy += best_eva_list[1]
            print(str(i) + '_Model evaluation: ', best_eva_list, "   Now ACC:",
                  str(round(avg_accuracy * 10000) / 100 / i))
            i += 1
            y_pred_best = self.model.predict(x[test])
            self.matrix.append(confusion_matrix(np.argmax(y[test], axis=1), np.argmax(y_pred_best, axis=1)))
            em = classification_report(np.argmax(y[test], axis=1), np.argmax(y_pred_best, axis=1),
                                       target_names=self.class_label, output_dict=True)
            self.eva_matrix.append(em)
            print(classification_report(np.argmax(y[test], axis=1), np.argmax(y_pred_best, axis=1),
                                        target_names=self.class_label))
            caps_layer_model = Model(inputs=self.model.input,
                                     outputs=self.model.get_layer(index=-2).output)
            feature_source = caps_layer_model.predict(x[test])
            x_feats.append(feature_source)
            y_labels.append(y[test])
        print("Average ACC:", avg_accuracy / self.args.split_fold)
        self.acc = avg_accuracy / self.args.split_fold
        return x_feats, y_labels
