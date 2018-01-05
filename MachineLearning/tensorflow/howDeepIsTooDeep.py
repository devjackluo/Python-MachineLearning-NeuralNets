import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
input  --weight> hidden layer 1 (activation function) --weight> hidden layer 2 (act func) --weight> output layer

compare output to intended output -> cost/loss function (cross entropy?)
optimization function (optimizer) -> minimize cost (adamOptimizer...SGD, AdaGrad??)

backpropagation

feed forward + backprop = epoch

'''

# mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)
# with open('/tmp/mnist.pickle', 'wb') as f:
#     pickle.dump(mnist, f)

pickle_in = open('/tmp/mnist.pickle', 'rb')
mnist = pickle.load(pickle_in)


'''
10 classes, 0-9

0 = 0
1 = 1...

ONE HOT (one in on, all off)
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]...

'''

# number of nodes for hidden layers (can be different for each layer)
n_nodes_hl1 = 500
n_nodes_hl2 = 200
n_nodes_hl3 = 500
n_nodes_hl4 = 200
n_nodes_hl5 = 500
n_nodes_hl6 = 200

n_classes = 10
# how much to do each time to not consume all the RAM
batch_size = 100

# 55000 images
#print(int(mnist.train.num_examples))

# height x width of object/image?
# x_ph = tf.placeholder('float', shape=[None, 784])
# y_ph = tf.placeholder('float', shape=[batch_size, n_classes])
# no shape, tensorflow will dynamically do it
x_ph = tf.placeholder('float')
y_ph = tf.placeholder('float')



def neural_network_model(data):

    # creates a link from the 784 starting nodes (28x28 image) which will have a random weight
    # also create a bias in the event of 0x0 so the neuron will still fire / better accuracy having two variables?
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    hidden_4_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl4]))}
    hidden_5_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl4, n_nodes_hl5])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl5]))}
    hidden_6_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl5, n_nodes_hl6])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl6]))}


    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl6, n_classes])), 'biases': tf.Variable(tf.random_normal([n_classes]))}



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

    l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
    l4 = tf.nn.relu(l4)
    l5 = tf.add(tf.matmul(l4, hidden_5_layer['weights']), hidden_5_layer['biases'])
    l5 = tf.nn.relu(l5)
    l6 = tf.add(tf.matmul(l5, hidden_6_layer['weights']), hidden_6_layer['biases'])
    l6 = tf.nn.relu(l6)



    output = tf.matmul(l6, output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x_data):

    # shoot our data into neural net and get a class [0-10] output value
    prediction = neural_network_model(x_data)

    # mean of difference between prediction vs y_placeholder
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_ph))

    # default learning rate = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    # cycles of feed + backprop
    hm_epochs = 10

    # create saver
    saver = tf.train.Saver()

    # run tensor session
    with tf.Session() as sess:

        # all tf.Variable are initialized
        sess.run(tf.global_variables_initializer())

        # loads old model
        try:
            saver.restore(sess, "/tmp/model.ckpt")
            print("Model Loaded")
        except:
            pass

        # for each cycle
        for epoch in range(hm_epochs):
            epoch_loss = 0
            # for each batch
            for _ in range(int(mnist.train.num_examples/batch_size)):

                # get each batches's value (pixel on/off) and get actual value (class)
                # epoch x is (28x28), epoch_y is (1x10)
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)

                # TensorFlow's AdamOptimizer tweaks the loss
                # Take note we're doing this every batch not every epoch
                _, c = sess.run([optimizer, loss], feed_dict={x_ph: epoch_x, y_ph: epoch_y})

                epoch_loss += c

            print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss', epoch_loss)

        # after training, we see how many it got correct and see how good it is vs test sample.

        # argmax returns the maximum of array (one hot so all 0s and one 1) and compare
        # return tensor of bool
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_ph, 1))
        # convert true/false to 1/0s
        # then take the reduced_mean tensor
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # then, compare test sample by evaluating it with accuracy tensor
        print('Accuracy:', accuracy.eval({x_ph: mnist.test.images, y_ph: mnist.test.labels}))


        userinput = input('To Save Model? : ').lower()
        if userinput == 'y':
            # saves new model
            save_path = saver.save(sess, "/tmp/model.ckpt")
            print("Model saved in file: %s" % save_path)

        # save_path = saver.save(sess, "/tmp/model.ckpt")
        # print("Model saved in file: %s" % save_path)


train_neural_network(x_ph)






