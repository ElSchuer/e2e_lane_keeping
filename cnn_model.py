import tensorflow as tf
import scipy

input_width = 200
input_height = 66
input_dim = 1

def create_weight_var(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def create_bias_var(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')



x = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_dim])
y_in = tf.placeholder(tf.float32, shape=[None, 1])

x_img = x

#######################################
# Convolution Layer 1-5
#######################################
k_num_1 = 24
W_conv1 = create_weight_var([5, 5, input_dim, k_num_1])
b_conv1 = create_bias_var([k_num_1])
conv1 = tf.nn.relu(conv2d(x_img, W_conv1, 2) + b_conv1)

k_num_2 = 36
W_conv2 = create_weight_var([5, 5, k_num_1, k_num_2])
b_conv2 = create_bias_var([k_num_2])
conv2 = tf.nn.relu(conv2d(conv1, W_conv2, 2) + b_conv2)

k_num_3 = 48
W_conv3 = create_weight_var([5, 5, k_num_2, k_num_3])
b_conv3 = create_bias_var([k_num_3])
conv3 = tf.nn.relu(conv2d(conv2, W_conv3, 2) + b_conv3)

k_num_4 = 64
W_conv4 = create_weight_var([3, 3, k_num_3, k_num_4])
b_conv4 = create_bias_var([k_num_4])
conv4 = tf.nn.relu(conv2d(conv3, W_conv4, 1) + b_conv4)

k_num_5 = 64
W_conv5 = create_weight_var([3, 3, k_num_4, k_num_5])
b_conv5 = create_bias_var([k_num_5])
conv5 = tf.nn.relu(conv2d(conv4, W_conv5, 1) + b_conv5)


#######################################
# Fully Connected Layer 6-9
#######################################


# Layer 6
W_fc1 = create_weight_var([1152, 1164])
b_fc1 = create_bias_var([1164])
conv5_flat = tf.reshape(conv5, [-1, 1152])
h_fc1 = tf.nn.relu(tf.matmul(conv5_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# Layer 7
W_fc2 = create_weight_var([1164, 100])
b_fc2 = create_bias_var([100])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# Layer 8
W_fc3 = create_weight_var([100, 50])
b_fc3 = create_bias_var([50])
h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

# Layer 9
W_fc4 = create_weight_var([50, 10])
b_fc4 = create_bias_var([10])
h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)
h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

#Output
W_fc5 = create_weight_var([10, 1])
b_fc5 = create_bias_var([1])

y = tf.multiply(tf.atan(tf.matmul(h_fc4_drop, W_fc5) + b_fc5), 2) #scale the atan output