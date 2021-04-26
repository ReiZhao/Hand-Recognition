# -*- coding: utf-8 -*-
"""
    具有早停保护的DNN网络训练函数，适用于静态手势识别，识别数据为手势特征提取得到的Hu矩数据库
    此文件调用了Tensorflow自带的DNNclassifeir 2019.04.21
"""
import argparse
import tensorflow as tf
import read_data
import os

parse = argparse.ArgumentParser()
# 添加参数batch_size
parse.add_argument('--batch_size', default=480, type=int,
                   help='batch size')
# 添加参数train_steps
parse.add_argument('--train_steps', default=50, type=int,
                   help='training steps')
# 添加参数train_steps
parse.add_argument('--train_final_steps', default=7000, type=int,
                   help='training final steps')
# 添加网络内容存储路径
parse.add_argument('--model_save_path', default="/Users/rei/PycharmProjects/DNN/DNN_message_Adam", type=str,
                   help='model_save_path')
model_save = "/Users/rei/PycharmProjects/DNN/DNN_message_Adam"


def remove(path):
    """
        删除该路径下的所有文件，但保留最上层文件夹
    """
    # args = parse.parse_args(argv[1:])
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    # os.rmdir(path)


def main(argv):
    """
        主函数
        构造并训练DNN网络，使用测试集测试正确率
        自动存储网络信息至规定文件夹内
    """
    # parse_args() 解析添加的参数
    args = parse.parse_args(argv[1:])

    (train_x, train_y), (test_x, test_y) = read_data.load_data()

    my_feature_columns = []
    # 按键值读入对应的 feature 进 my_feature_columns
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    """
        函数名称：DNN分类器

        model_dir: 训练后网络的存储地址
        hidden_units: 隐含层每层的结点数
        n_classes：分类数
        optimizer：梯度下降算法--Adam优化算法  
        activation_fn：层间激活函数，使用常用的relu函数
        dropout：学习速率递减率，不启用
    """
    # 调用高级API实现DNN分类器架构
    classifier = tf.estimator.DNNClassifier(
        model_dir=args.model_save_path,
        feature_columns=my_feature_columns,
        hidden_units=[14, 29, 14],
        n_classes=6,
        activation_fn=tf.nn.relu,
        dropout=None,
        optimizer='Adam'
    )
    """
    训练DNN网络
    input_fn：使用编写的喂食函数读入训练用样本  每次读入的样本数使用预设值：args.batch_size
    #·······································训练次数使用预设值：args.train_steps
    count 记录总迭代次数
    每50次循环训练进行一次模型评估，直至模型预测准确度达到预设要求 
    """
    iters = 0
    while True:
        classifier.train(
            input_fn=lambda: read_data.train_input_fn(train_x, train_y, args.batch_size),
            steps=args.train_steps
        )

        # 测试DNN网络准确性
        eval_result = classifier.evaluate(
            input_fn=lambda: read_data.eval_input_fn(test_x, test_y, args.batch_size)
        )

        if eval_result['accuracy'] >= 0.94:
            break

        iters = iters + args.train_steps
        print("iters:", iters)
        print("accuracy:", eval_result['accuracy'])

        if iters >= args.train_final_steps:
            break

    # 按{x}标记位置输出数据，只输出结果的第一项准确度的两位近似小数
    print('\nTest set accuracy: {accuracy:0.2f}\n'.format(**eval_result))
    print('\nNetwork test message: {}\n'.format(eval_result))

    # 额外验证阶段
    expected = ['middle_finger', 'six', 'good', 'two']
    # 创建一个待测字典，字典可以接受变量输入
    predict_x = {
        'Hu1': [0.245876844736, 0.202608213, 0.201082394, 0.222258981],
        'Hu2': [0.030623850201, 0.000932818, 0.007245985, 0.015059035],
        'Hu3': [0.001013530044, 0.004525317, 0.00241152, 0.000663658],
        'Hu4': [0.000312775227, 9.82E-05, 0.000114731, 0.000611472],
        'Hu5': [0.000000169638, 2.93E-08, -2.76E-08, 3.89E-07],
        'Hu6': [0.000035861528, 2.90E-06, -8.71E-06, 7.49E-05],
        'Hu7': [0.000000047278, 5.85E-08, -5.36E-08, 1.98E-08]
    }

    # 使用训练得到的网络针对新样本进行分类,顺序遍历所有需要预测的数据
    predictions = classifier.predict(
        input_fn=lambda: read_data.eval_input_fn(predict_x,
                                                 labels=None,
                                                 batch_size=1)
    )
    # 暂存输出信息
    template = ('\nPrediction is "{}" ({:.1f}%) , expected "{}"')

    for pre_dict, expec in zip(predictions, expected):
        class_id = pre_dict['class_ids'][0]
        probability = pre_dict['probabilities'][class_id]
        print(template.format(read_data.SPECIES[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    # 重新训练前先清除自动保存的网络信息
    remove(model_save)

    # 输出 INFO Warning两级的日志信息
    # tf.logging.set_verbosity(tf.logging.INFO)

    # tf定义了tf.app.flags，用于支持接受命令行传递参数，相当于接受argv。
    # 且此方法调用了高级API，所有网络需要的参数均带有默认值
    # 所以此处不需要再初始化参数，直接app.run即可
    tf.app.run(main)
