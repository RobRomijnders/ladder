# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:47:31 2016

@author: rob

Code base from
https://github.com/rinuboney/ladder

- Why start backpropping from the softmaxed-layer?

"""

import tensorflow as tf
from tensorflow.python import control_flow_ops
import input_data
import math
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

channel_sizes = [1,12,10,10]  #Last layer will by fully conv
denoising_cost =10* np.array([10.0, 10.0,0.1, 0.10])


logsave = False   # Do you want log files and checkpoint savers?
vis = True        #Visualize the Original - Noised - Recovered for the unsupervised samples

C = len(channel_sizes) - 1 # number of channel sizes

map_sizes = [28]
for c in range(1,C):
  map_sizes += [map_sizes[c-1]-2]

num_examples = 60000
num_epochs = 150
num_labeled = 100
num_classes  =10

starter_learning_rate = 0.02

decay_after = 15  # epoch after which to begin learning rate decay

batch_size = 100
num_iter = (num_examples/batch_size) * num_epochs  # number of loop iterations

inputs = tf.placeholder(tf.float32, shape=(None, 784))
images = tf.reshape(inputs,[-1,28,28,1])
outputs = tf.placeholder(tf.float32)


def bi(inits, size, name):
  return tf.Variable(inits * tf.ones([size]), name=name)


def wi(shape, name):
  return tf.Variable(tf.random_normal(shape, name=name)) / math.sqrt(shape[0])


weights = {'W': [0]*C,
           'V': [0]*C,
           # batch normalization parameter to shift the normalized value
           'beta': [bi(0.0, channel_sizes[l+1], "beta") for l in range(C)],
           # batch normalization parameter to scale the normalized value
           'gamma': [bi(1.0, channel_sizes[l+1], "beta") for l in range(C)]}

initi = tf.uniform_unit_scaling_initializer(factor=1.43)
for c in range(C):
  if c == C-1: #Make the kernel as big as the final map sizes. Similar to Fully Convolutional Layer
    width = map_sizes[-1]
    shape = [width,width,channel_sizes[c],channel_sizes[c+1]]
    weights['W'][c] = tf.get_variable(name='W'+str(c), shape=shape, initializer = initi)
  else:
    shape = [3,3,channel_sizes[c],channel_sizes[c+1]]
    weights['W'][c] = tf.get_variable(name='W'+str(c), shape=shape, initializer = initi)
  print('W%s has shape '%c+str(shape))
for c in range(C-1,-1,-1):
  if c == C-1:
    width = map_sizes[-1]
    shape = [width,width,channel_sizes[c],channel_sizes[c+1]]
    weights['V'][c] = tf.get_variable(name='V'+str(c), shape=shape, initializer = initi)
  else:
    shape = [3,3,channel_sizes[c],channel_sizes[c+1]]
    weights['V'][c] = tf.get_variable(name='V'+str(c), shape=shape, initializer = initi)
  print('V%s has shape '%c+str(shape))



noise_std = 0.3  # scaling factor for noise used in corrupted encoder

# hyperparameters that denote the importance of each layer

#Note, these four functions work now for 4D Tensors
join = lambda l, u: tf.concat(0, [l, u])
labeled = lambda x: tf.slice(x, [0, 0,0,0], [batch_size, -1,-1,-1]) if x is not None else x
unlabeled = lambda x: tf.slice(x, [batch_size, 0,0,0], [-1, -1,-1,-1]) if x is not None else x
split_lu = lambda x: (labeled(x), unlabeled(x))
#The old functions that work with 2D Tensors
labeled2 = lambda x: tf.slice(x, [0, 0], [batch_size, -1]) if x is not None else x
unlabeled2 = lambda x: tf.slice(x, [batch_size, 0], [-1, -1]) if x is not None else x

training = tf.placeholder(tf.bool)

ewma = tf.train.ExponentialMovingAverage(decay=0.99)  # to calculate the moving averages of mean and variance
bn_assigns = []  # this list stores the updates to be made to average mean and variance


def batch_normalization(batch, mean=None, var=None,axes=[0,1,2]):
  """Set axes to [0] for batch-norm in a 2D Tensor"""
  if mean is None or var is None:
      mean, var = tf.nn.moments(batch, axes=axes)
  return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))


# average mean and variance of all layers
running_mean = [tf.Variable(tf.constant(0.0, shape=[c]), trainable=False) for c in channel_sizes[1:]]
running_var = [tf.Variable(tf.constant(1.0, shape=[c]), trainable=False) for c in channel_sizes[1:]]


def update_batch_normalization(batch, l,axes=[0,1,2]):
  "batch normalize + update average mean and variance of layer l"
  mean, var = tf.nn.moments(batch, axes=axes)
  assign_mean = running_mean[l-1].assign(mean)
  assign_var = running_var[l-1].assign(var)
  bn_assigns.append(ewma.apply([running_mean[l-1], running_var[l-1]]))
  with tf.control_dependencies([assign_mean, assign_var]):
    return (batch - mean) / tf.sqrt(var + 1e-10)


def encoder(images, noise_std):
  h = images + tf.random_normal(tf.shape(images)) * noise_std  # add noise to input
  d = {}  # to store the pre-activation, activation, mean and variance for each layer
  # The data for labeled and unlabeled examples are stored separately
  d['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
  d['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
  d['labeled']['z'][0], d['unlabeled']['z'][0] = split_lu(h)
  for l in range(1, C+1):
    print "Layer ", l, ": ", channel_sizes[l-1], " -> ", channel_sizes[l]
    d['labeled']['h'][l-1], d['unlabeled']['h'][l-1] = split_lu(h)
#    z_pre = tf.matmul(h, weights['W'][l-1])  # pre-activation
    z_pre = tf.nn.conv2d(h, weights['W'][l-1], strides=[1, 1, 1, 1], padding='VALID')
    z_pre_l, z_pre_u = split_lu(z_pre)  # split labeled and unlabeled examples

    m, v = tf.nn.moments(z_pre_u, axes=[0,1,2]) #in size [,channel_sizes[l]]

    # if training:
    def training_batch_norm():
      # Training batch normalization
      # batch normalization for labeled and unlabeled examples is performed separately
      if noise_std > 0:
        # Corrupted encoder
        # batch normalization + noise
        z = join(batch_normalization(z_pre_l), batch_normalization(z_pre_u, m, v))
        z += tf.random_normal(tf.shape(z_pre)) * noise_std
      else:
        # Clean encoder
        # batch normalization + update the average mean and variance using batch mean and variance of labeled examples
        z = join(update_batch_normalization(z_pre_l, l), batch_normalization(z_pre_u, m, v))
      return z

    # else:
    def eval_batch_norm():
      # Evaluation batch normalization
      # obtain average mean and variance and use it to normalize the batch
      mean = ewma.average(running_mean[l-1])
      var = ewma.average(running_var[l-1])
      z = batch_normalization(z_pre, mean, var)
      # Instead of the above statement, the use of the following 2 statements containing a typo
      # consistently produces a 0.2% higher accuracy for unclear reasons.
      # m_l, v_l = tf.nn.moments(z_pre_l, axes=[0])
      # z = join(batch_normalization(z_pre_l, m_l, mean, var), batch_normalization(z_pre_u, mean, var))
      return z

    # perform batch normalization according to value of boolean "training" placeholder:
    z = control_flow_ops.cond(training, training_batch_norm, eval_batch_norm)
    if l == C:
      # use softmax activation in output layer
      h = tf.nn.softmax(weights['gamma'][l-1] * (tf.squeeze(z,squeeze_dims=[1,2]) + weights["beta"][l-1]))
      h = tf.expand_dims(tf.expand_dims(h,dim=1),dim=2)
    else:
      # use ReLU activation in hidden layers
      h = tf.nn.relu(z + weights["beta"][l-1])
    d['labeled']['z'][l], d['unlabeled']['z'][l] = split_lu(z)
    d['unlabeled']['m'][l], d['unlabeled']['v'][l] = m, v  # save mean and variance of unlabeled examples for decoding
  d['labeled']['h'][l], d['unlabeled']['h'][l] = split_lu(h)
  return h, d

print "=== Corrupted Encoder ==="
y_c, corrupted = encoder(images, noise_std)

print "=== Clean Encoder ==="
y, clean = encoder(images, 0.0)  # 0.0 -> do not add noise

print "=== Decoder ==="


def g_gauss(z_c, u, size):
  """gaussian denoising function proposed in the original paper
  z_c: corrupted latent variable
  u: Tensor from layer (l+1) in decorder
  size: number hidden neurons for this layer"""
  wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)
  a1 = wi(0., 'a1')
  a2 = wi(1., 'a2')
  a3 = wi(0., 'a3')
  a4 = wi(0., 'a4')
  a5 = wi(0., 'a5')

  a6 = wi(0., 'a6')
  a7 = wi(1., 'a7')
  a8 = wi(0., 'a8')
  a9 = wi(0., 'a9')
  a10 = wi(0., 'a10')
  #Crazy transformation of the prior (mu) and convex-combi weight (v)
  mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5  #prior
  v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10  #convex-combi weight

  z_est = (z_c - mu) * v + mu  #equation [2] in http://arxiv.org/pdf/1507.02672v2.pdf
  return z_est

# Decoder
z_est = {}
d_cost = []  # to store the denoising cost of all layers
for l in range(C, -1, -1):
  print "Layer ", l, ": ", channel_sizes[l+1] if l+1 < len(channel_sizes) else None, " -> ", channel_sizes[l], ", denoising cost: ", denoising_cost[l]
  z, z_c = clean['unlabeled']['z'][l], corrupted['unlabeled']['z'][l]
  m, v = clean['unlabeled']['m'].get(l, 0), clean['unlabeled']['v'].get(l, 1-1e-10)
  # m are batch-norm means, v are batch-norm stddevs
  if l == C:
    u = unlabeled(y_c)
  else:
    u = tf.nn.conv2d_transpose(z_est[l+1], weights['V'][l], tf.pack([tf.shape(z_est[l+1])[0], map_sizes[l], map_sizes[l], channel_sizes[l]]),strides=[1, 1, 1, 1], padding='VALID',name = 'CT'+str(l))
  u = batch_normalization(u)
  z_est[l] = g_gauss(z_c, u, channel_sizes[l])
  z_est_bn = (z_est[l] - m) / v
  # append the cost of this layer to d_cost
  d_cost.append((tf.reduce_mean(tf.reduce_sum(tf.square(z_est_bn - z), 1)) / channel_sizes[l]) * denoising_cost[l])

# calculate total unsupervised cost by adding the denoising cost of all layers
u_cost = tf.add_n(d_cost)

y_N = labeled(y_c)

#Convert y* back to 2D Tensor
y_N = tf.squeeze(y_N, squeeze_dims=[1,2])
y = tf.squeeze(y, squeeze_dims=[1,2])

s_cost = -tf.reduce_mean(tf.reduce_sum(outputs*tf.log(y_N), 1))  # supervised cost
loss = s_cost + u_cost  # total cost

#pred_cost = -tf.reduce_mean(tf.reduce_sum(outputs*tf.log(y), 1))  # cost used for prediction
correct_prediction = tf.equal(tf.argmax(labeled2(y), 1), tf.argmax(outputs, 1))  # no of correct predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * tf.constant(100.0)

learning_rate = tf.Variable(starter_learning_rate, trainable=False)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# add the updates of batch normalization statistics to train_step
bn_updates = tf.group(*bn_assigns)
with tf.control_dependencies([train_step]):
    train_step = tf.group(bn_updates)

print "===  Loading Data ==="
mnist = input_data.read_data_sets("MNIST_data", n_labeled=num_labeled, one_hot=True, val_ratio = 0.15)

if logsave: saver = tf.train.Saver()

print "===  Starting Session ==="
sess = tf.Session()

i_iter = 0

ckpt = tf.train.get_checkpoint_state('checkpoints/')  # get latest checkpoint (if any)
if ckpt and ckpt.model_checkpoint_path and logsave:
  # if checkpoint exists, restore the parameters and set epoch_n and i_iter
  saver.restore(sess, ckpt.model_checkpoint_path)
  epoch_n = int(ckpt.model_checkpoint_path.split('-')[1])
  i_iter = (epoch_n+1) * (num_examples/batch_size)
  print "Restored Epoch ", epoch_n
else:
  # no checkpoint exists. create checkpoints directory if it does not exist.
  if not os.path.exists('checkpoints'):
      os.makedirs('checkpoints')
  init = tf.initialize_all_variables()
  sess.run(init)

print "=== Training ==="
#print "Initial Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels, training: False}), "%"
acc_ma = 0.0
s_cost_ma = 0.0
u_cost_ma = 0.0
for i in range(i_iter, num_iter):
  images, labels = mnist.train.next_batch(batch_size)
  result = sess.run([train_step,accuracy,s_cost,u_cost], feed_dict={inputs: images, outputs: labels, training: True})
  acc_ma = 0.8*acc_ma+0.2*result[1]
  s_cost_ma = 0.8*s_cost_ma+0.2*result[2]
  u_cost_ma = 0.8*u_cost_ma+0.2*result[3]

#  print(debug)
  if (i > 1) and i%10 == 0:  #((i+1) % (num_iter/num_epochs) == 0)
    epoch_n = i/(num_examples/batch_size)
    if (epoch_n+1) >= decay_after:
      # decay learning rate
      # learning_rate = starter_learning_rate * ((num_epochs - epoch_n) / (num_epochs - decay_after))
      ratio = 1.0 * (num_epochs - (epoch_n+1))  # epoch_n + 1 because learning rate is set for next epoch
      ratio = max(0, ratio / (num_epochs - decay_after))
      sess.run(learning_rate.assign(starter_learning_rate * ratio))
    if logsave: saver.save(sess, 'checkpoints/model.ckpt', epoch_n)
    fetch = [accuracy,s_cost,u_cost]
    if vis: fetch += [corrupted['unlabeled']['z'][0],z_est[0]]
    images_val, labels_val = mnist.validation.next_batch(batch_size)
    result = sess.run(fetch, feed_dict={inputs: images_val, outputs:labels_val, training: False})
    print("At %5.0f of %5.0f acc %5.1f(%5.1f) cost super %6.2f(%6.2f) unsuper %6.2f(%6.2f)"%(i,num_iter,result[0],acc_ma,result[1],s_cost_ma,result[2],u_cost_ma))

    #Visualize
    if vis and i%100 == 0:
      Nplot = 3
      ind = np.random.choice(num_labeled,Nplot)
      f, axarr = plt.subplots(Nplot, 3)
      for r in range(Nplot):
        axarr[r, 0].imshow(np.reshape(images_val[batch_size+ind[r]],(28,28)), cmap=plt.get_cmap('gray'),vmin = 0.0, vmax=1.0)
        axarr[r, 1].imshow(np.reshape(result[3][ind[r]],(28,28)), cmap=plt.get_cmap('gray'),vmin = 0.0, vmax=1.0)
        axarr[r, 2].imshow(np.reshape(result[4][ind[r]],(28,28)), cmap=plt.get_cmap('gray'),vmin = 0.0, vmax=1.0)
        plt.setp([a.get_xticklabels() for a in axarr[r, :]], visible=False)
        plt.setp([a.get_yticklabels() for a in axarr[r, :]], visible=False)
      f.subplots_adjust(wspace=0.0, hspace = 0.0)
      f.suptitle('Original - Corrupted - Recovered')
      plt.show()


    if logsave:
      with open('train_log', 'ab') as train_log:
        # write test accuracy to file "train_log"
        train_log_w = csv.writer(train_log)
        log_i = [epoch_n] + sess.run([accuracy], feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels, training: False})
        train_log_w.writerow(log_i)

print "Final Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels, training: False}), "%"

sess.close()