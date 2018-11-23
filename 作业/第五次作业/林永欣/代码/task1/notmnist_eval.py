import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pickle  # 它可以将对象转换为一种可以传输或存储的格式 序列化过程将文本信息转变为二进制数据流
import notmnist_inference
import notmnist_train
# import notmnist_data

# 加载的时间间隔。
EVAL_INTERVAL_SECS = 10


def evaluate(valid_features, valid_labels, test_features, test_labels):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, notmnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, notmnist_inference.OUTPUT_NODE], name='y-input')
        validate_feed = {x: valid_features, y_: valid_labels}
        test_feed = {x: test_features, y_: test_labels}

        y = notmnist_inference.inference(x, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(notmnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(notmnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    pickle_file = r'D:\中科大软院\数据集\notMNIST.pickle'
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
        valid_features = pickle_data['valid_dataset']
        valid_labels = pickle_data['valid_labels']
        test_features = pickle_data['test_dataset']
        test_labels = pickle_data['test_labels']
        del pickle_data  # Free up memory

    print('Data and modules loaded.')
    evaluate(valid_features, valid_labels, test_features, test_labels)


if __name__ == '__main__':
    main()

