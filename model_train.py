# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, BatchNormalization, Dense
from bert.extract_feature import BertVector
import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.metrics as sm
import sklearn.ensemble as es
import sklearn.naive_bayes as nb


def embedding(bert_model, ind):
    df = pd.read_csv('data/train_data%d.csv'%(ind), encoding='gbk')
    df = df.drop_duplicates()
    df['岗位职责'] = df['岗位名称'] + df['岗位职责']
    df = df.drop('岗位名称', axis=1).dropna()
    df.columns = ['text', 'label']
    print(df.head())
    # 读取文件并进行转换
    print('begin encoding')
    f = lambda text: bert_model.encode([text])["encodes"][0]
    df['x'] = df['text'].apply(f)
    print('end encoding')

    x_df = np.array([vec for vec in df['x']])
    y_df = np.array([vec for vec in df['label']])
    np.save('data/x_df%d.npy'%(ind),x_df)
    np.save('data/y_df%d.npy'%(ind),y_df)


def load_split(ind):
    x_df = np.load('data/x_df%d.npy'%(ind))
    y_df = np.load('data/y_df%d.npy'%(ind))*1
    x_train, x_test, y_train, y_test \
        = ms.train_test_split(x_df, y_df, test_size=0.3, random_state=1)
    return x_train, x_test, y_train, y_test


def skl_precision(algo, x_train, x_test, y_train, y_test, **kwargs):
    '''

    :param algo: sklearn库内的模型
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :param kwargs: algo模型的超参数
    :return: 训练好的模型
    '''
    model = algo(**kwargs)
    print(model)
    model.fit(x_train, y_train)
    pred_test_y = model.predict(x_test)
    bg = sm.classification_report(y_test, pred_test_y)
    print('分类报告：', bg, sep='\n')
    return model


def mlp_precision(x_train, x_test, y_train, y_test):
    num_classes = 2
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    x_in = Input(shape=(768, ))
    x_out = Dense(32, activation="relu")(x_in)
    x_out = BatchNormalization()(x_out)
    x_out = Dense(num_classes, activation="softmax")(x_out)
    model = Model(inputs=x_in, outputs=x_out)
    #print(model.summary())
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=10, epochs=20)
    model.evaluate(x_test, y_test)
    return model


def test_model(bert_model, models):
    test_data = pd.read_csv('data/sample.csv',encoding='gbk')
    test = pd.DataFrame(test_data, columns=['岗位名称', '岗位职责'])
    test['text'] = test['岗位名称'] + test['岗位职责']
    print('begin encoding')
    f = lambda text: bert_model.encode([text])["encodes"][0]
    test['x'] = test['text'].apply(f)
    print('end encoding')
    pred_data = np.array([_ for _ in test['x']])
    for ind, model in enumerate(models):
        if ind != 3:
            pred = model.predict(pred_data)
            test['pred%d'%(ind)] = pred
        else:
            pred = model.predict(pred_data)
            y = np.argmax(pred, axis=1)
            test['pred%d'%(ind)] = y
    test = test.drop('x', axis=1)
    test.to_csv('result/output.csv',encoding='gbk')


if __name__ == '__main__':
    bert_model = BertVector(pooling_strategy="REDUCE_MEAN", max_seq_len=400)
    #embedding(bert_model, 2) #只需在更换训练数据时执行，得到的词向量将保存在data下，无需重复执行
    data = load_split(2)
    #embedding与load_split的参数与训练数据文件名的末尾数字保持一致
    model1 = skl_precision(svm.SVC, *data, kernel='rbf')
    model2 = skl_precision(es.RandomForestClassifier, *data, max_features='sqrt')
    model3 = skl_precision(nb.GaussianNB, *data)
    model4 = mlp_precision(*data)
    test_model(bert_model, [model1, model2, model3, model4])
