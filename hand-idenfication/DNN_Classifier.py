# -*- coding: utf-8 -*-

"""
    含有三层隐含层的DNN网络，适用于静态手势识别，识别数据为手势特征提取得到的Hu矩数据库
    此文件直接调用已训练好的DNN网络，若载入新的DNN网络，必须同步更改此函数下classifier的参数 2019.04.21
"""
import cv2
import tensorflow as tf
import read_data
# 训练好的模型载入地址
model_load = "DNN_message_Adam"

# 复载入训练用数据，防止tensoflow检查点WARNING
(train_x, train_y), (test_x, test_y) = read_data.load_data()
my_feature_columns = []
# 按键值读入对应的 feature 进 my_feature_columns
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

# 调用高级API实现DNN分类器架构
"""
    函数名称：DNN分类器

    model_dir: 训练后网络的存储地址
    hidden_units: 隐含层每层的结点数
    n_classes：分类数
    optimizer：梯度下降算法--Adam优化算法  
    activation_fn：层间激活函数，使用常用的relu函数
    dropout：学习速率递减率，不启用
"""
classifier = tf.estimator.DNNClassifier(
    model_dir=model_load,
    feature_columns=my_feature_columns,
    hidden_units=[28, 29, 16],
    n_classes=8,
    activation_fn=tf.nn.relu,
    dropout=None,
    optimizer='Adam'
)


def Predict(Huments):
    """
        函数名称：DNN分类器预测算法
        input：
            七纬Hu矩

        predict_x：带预测的样本
        predictions：预测信息汇总

        output：
            class_id：预测的标签编码值
            specie：预测的标签英文名

    """
    # 创建一个待测字典，字典可以接受变量输入
    predict_x = {
        'Hu1': Huments[0],
        'Hu2': Huments[1],
        'Hu3': Huments[2],
        'Hu4': Huments[3],
        'Hu5': Huments[4],
        'Hu6': Huments[5],
        'Hu7': Huments[6]
    }

    # 使用训练得到的网络针对新样本进行分类,顺序遍历所有需要预测的数据
    predictions = classifier.predict(
        input_fn=lambda: read_data.eval_input_fn(predict_x,
                                                 labels=None,
                                                 batch_size=1)
    )
    # 暂存输出信息
    template = ('------Prediction is "{}" Probabilities: ({:.1f}%)\n')
    for pre_dict in predictions:
        class_id = pre_dict['class_ids'][0]
        probability = pre_dict['probabilities'][class_id]
        specie = read_data.SPECIES[class_id]
        # 预测概率超过半数才输出预测结果
        if probability >= 0.5:
            print(template.format(specie, 100 * probability))
        # else:
            # print('--------Prediction  Defeat---------\n')
        if probability >= 0.5:
            # 返回预测样本标签
            return class_id, specie
        else:
            # 返回未使用的标记8
            return 8, "None"


if __name__ == '__main__':
    # 测试用Huments
    Huments = [[0.245876844736],
              [0.030623850201],
              [0.001013530044],
              [0.000312775227],
              [0.000000169638],
              [0.000035861528],
              [0.000000047278]]
    Predict(Huments)
    # class_id, specie = Predict(Huments)
