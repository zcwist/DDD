import tensorflow as tf 

class Siamese:

    # Create model
    def __init__(self):
        self.x1 = tf.placeholder(tf.float32, [None, 200])
        self.x2 = tf.placeholder(tf.float32, [None, 200])
        self.y = tf.placeholder(tf.float32, [None])
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

        with tf.variable_scope("siamese") as scope:
            self.o1 = self.network(self.x1)
            scope.reuse_variables()
            self.o2 = self.network(self.x2)

        # Create loss
        self.loss = self.logistic_loss()

    def network(self, x):
        weights = []
        fc1 = self.fc_layer(x, 512, "fc1")
        ac1 = tf.nn.tanh(fc1)
        ac1 = tf.nn.dropout(ac1, self.keep_prob)
        fc2 = self.fc_layer(ac1, 512, "fc2")
        ac2 = tf.nn.tanh(fc2)
        ac2 = tf.nn.dropout(ac2, self.keep_prob)
        fc3 = self.fc_layer(ac2, 2, "fc3")
        ac3 = tf.nn.tanh(fc3)
        return ac3

    def fc_layer(self, bottom, n_weight, name):
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.1)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.001, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc

    def logistic_loss(self):

        diff = tf.reduce_sum(tf.abs(tf.sub(self.o1, self.o2)), 1)
        loss = tf.reduce_mean(tf.abs(self.y - diff), name="loss")

        # eucd2 = tf.pow(tf.sub(self.o1, self.o2), 2)
        # eucd2 = tf.reduce_sum(eucd2, 1)
        # eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        # loss = tf.reduce_mean(self.y-eucd, name="loss")
        return loss


    def loss_with_spring(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.sub(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.sub(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
        pos = tf.mul(labels_t, eucd2, name="yi_x_eucd2")
        # neg = tf.mul(labels_f, tf.sub(0.0,eucd2), name="yi_x_eucd2")
        # neg = tf.mul(labels_f, tf.maximum(0.0, tf.sub(C,eucd2)), name="Nyi_x_C-eucd_xx_2")
        neg = tf.mul(labels_f, tf.pow(tf.maximum(tf.sub(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss

    def loss_with_step(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.sub(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.sub(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        pos = tf.mul(labels_t, eucd, name="y_x_eucd")
        neg = tf.mul(labels_f, tf.maximum(0.0, tf.sub(C, eucd)), name="Ny_C-eucd")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss