"""
   前言：命名报错——
         TensorFlow 对 Tensor 和计算使用一个叫 name 的字符串辨识器，如果没有定义 name，TensorFlow 会自动创建一个；
         TensorFlow 会把第一个节点命名为 <Type>，把后续的命名为<Type>_<number>。
"""
import tensorflow as tf

"""报错示范
# 移除先前的权重和偏置项
tf.reset_default_graph()

save_file = 'model.ckpt'

# 两个 Tensor 变量：权重和偏置项
weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

saver = tf.train.Saver()

# 打印权重和偏置项的名字
print('Save Weights: {}'.format(weights.name))
print('Save Bias: {}'.format(bias.name))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, save_file)

# 移除之前的权重和偏置项
tf.reset_default_graph()

# 两个变量：权重和偏置项
bias = tf.Variable(tf.truncated_normal([3]))
weights = tf.Variable(tf.truncated_normal([2, 3]))

saver = tf.train.Saver()

# 打印权重和偏置项的名字
print('Load Weights: {}'.format(weights.name))
print('Load Bias: {}'.format(bias.name))

with tf.Session() as sess:
    # 加载权重和偏置项 - 报错
    saver.restore(sess, save_file)
"""

"""
   原因：weights 和 bias 的 name 属性与你保存的模型不同，所以
         会报“Assign requires shapes of both tensors to match”错误；
         saver.restore(sess, save_file) 代码试图把权重数据加载到bias里，把偏置项数据加载到 weights里。
"""

"""
   解决方案：手动设定name属性
"""
tf.reset_default_graph()

save_file = 'model.ckpt'

# 两个 Tensor 变量：权重和偏置项
weights = tf.Variable(tf.truncated_normal([2, 3]), name='weights_0')
bias = tf.Variable(tf.truncated_normal([3]), name='bias_0')

saver = tf.train.Saver()

# 打印权重和偏置项的名称
print('Save Weights: {}'.format(weights.name))
print('Save Bias: {}'.format(bias.name))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, save_file)

# 移除之前的权重和偏置项
tf.reset_default_graph()

# 两个变量：权重和偏置项
bias = tf.Variable(tf.truncated_normal([3]), name='bias_0')
weights = tf.Variable(tf.truncated_normal([2, 3]) ,name='weights_0')

saver = tf.train.Saver()

# 打印权重和偏置项的名称
print('Load Weights: {}'.format(weights.name))
print('Load Bias: {}'.format(bias.name))

with tf.Session() as sess:
    # 加载权重和偏置项 - 没有报错
    saver.restore(sess, save_file)

print('Loaded Weights and Bias successfully.')