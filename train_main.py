# python3
# -*- coding:utf-8 -*-
from load_model import Load_model
from load_batch import Load_batch
import tensorflow as tf
import numpy as np

def train(the_labels,model_names):
    model = Load_model(label=the_labels,numDimensions=300,model_name=model_names)
    train_data = Load_batch(method='train', label=the_labels,batchSize=128,numDimensions=300)
    validation_data = Load_batch(method='validation', label=the_labels,batchSize=128,numDimensions=300)

    graph = model.graph

    print("%s 图构建完成"%(the_labels))

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        optimizer = graph.get_operation_by_name('Optimise/Adam')
        input_data = graph.get_tensor_by_name('Inputs/input_data:0')
        input_length = graph.get_tensor_by_name('Inputs/input_length:0')
        labels = graph.get_tensor_by_name('Inputs/labels:0')
        merged = graph.get_tensor_by_name('Prediction/Merge/MergeSummary:0')
        # test
        prediction = graph.get_tensor_by_name('Prediction/ArgMax:0')
        confusion_matrix = graph.get_tensor_by_name('Prediction/confusion_matrix/SparseTensorDenseAdd:0')
        path_train = model.model_path + str("train")
        path_test = model.model_path + str("test")
        writer_train = tf.summary.FileWriter(logdir=path_train, graph=sess.graph)
        writer_test = tf.summary.FileWriter(logdir=path_test)
        saver = tf.train.Saver(max_to_keep=1)
        for i in range(4000 + 1):
            segMatrix, segLen, labelVec = train_data.next()
            feed_dict = {input_data: segMatrix, input_length: segLen, labels: labelVec}
            sess.run([optimizer], feed_dict=feed_dict)

            if i % 10 == 0 and i != 0:
                summary_train = sess.run(merged, feed_dict=feed_dict)
                writer_train.add_summary(summary_train, i)

                vsegMatrix, vsegLen, vlabelVec = validation_data.next()
                feed_dict_validation = {input_data: vsegMatrix, input_length: vsegLen, labels: vlabelVec}
                summary_test = sess.run(merged, feed_dict=feed_dict_validation)
                writer_test.add_summary(summary_test, i)
                print('addsummary_train...%s '%(the_labels) + str(i))
            if i % 200 == 0 and i != 0:
                path=model.model_path
                save_path = saver.save(sess, path, global_step=i)
                print('save_train...%s ' % (the_labels) + str(i))
            print('train...%s '%(the_labels) + str(i))
        writer_train.close()
        writer_test.close()
    # 开始测试
if __name__ == '__main__':
    model_name=["BiLSTM_selfattention","LSTM","selfattention"]
    for name in model_name:
        labels=np.arange(20)
        for i in labels:
            train(the_labels=i,model_names=name)