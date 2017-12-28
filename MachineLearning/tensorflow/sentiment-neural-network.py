import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
input  --weight> hidden layer 1 (activation function) --weight> hidden layer 2 (act func) --weight> output layer

compare output to intended output -> cost/loss function (cross entropy?)
optimization function (optimizer) -> minimize cost (adamOptimizer...SGD, AdaGrad??)

backpropagation

feed forward + backprop = epoch

'''
#mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)
'''
10 classes, 0-9

0 = 0
1 = 1...

ONE HOT (one in on, all off)
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]...

'''


# create our own features
from create_sentiment_featuresets import create_feature_sets_and_labels
# import pickle
# pickle_in = open('sentiment_set.pickle', 'rb')
# train_x, train_y, test_x, test_y = pickle.load(pickle_in)
train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')


print(len(train_x))


# number of nodes for hidden layers (can be different for each layer)
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500


n_classes = 2
# how much to do each time to not consume all the RAM
batch_size = 50


x_ph = tf.placeholder('float', [None, len(train_x[0])])
y_ph = tf.placeholder('float')



def neural_network_model(data):

    # creates a link from the 784 starting nodes (28x28 image) which will have a random weight
    # also create a bias in the event of 0x0 so the neuron will still fire / better accuracy having two variables?
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}


    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 'biases': tf.Variable(tf.random_normal([n_classes]))}


    # (input_data * weights) + biases
    # each piece of info from each is passed to the next layer
    # and some function (tf.nn.relu) is called to create new values
    # TensorFlow remembers these in memory? Build the layers
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)


    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']


    return output


def train_neural_network(x_data):

    # shoot our data into neural net and get a class [0-10] output value
    prediction = neural_network_model(x_data)

    # mean of difference between prediction vs y_placeholder
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_ph))

    # default learning rate = 0.001
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    # cycles of feed + backprop
    hm_epochs = 10


    # run tensor session
    with tf.Session() as sess:

        # all tf.Variable are initialized
        sess.run(tf.global_variables_initializer())

        # for each cycle
        for epoch in range(hm_epochs):
            epoch_loss = 0
            # for each batch


            #############################
            #############################
            # create our own batching system.
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                # TensorFlow's AdamOptimizer tweaks the loss
                # Take note we're doing this every batch not every epoch
                _, c = sess.run([optimizer, loss], feed_dict={x_ph: batch_x, y_ph: batch_y})

                epoch_loss += c

                # increment i
                i += batch_size

            print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss', epoch_loss)

        # after training, we see how many it got correct and see how good it is vs test sample.

        # argmax returns the maximum of array (one hot so all 0s and one 1) and compare
        # return tensor of bool
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_ph, 1))
        # convert true/false to 1/0s
        # then take the reduced_mean tensor
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # then, compare test sample by evaluating it with accuracy tensor
        print('Accuracy:', accuracy.eval({x_ph: test_x, y_ph: test_y}))


train_neural_network(x_ph)






