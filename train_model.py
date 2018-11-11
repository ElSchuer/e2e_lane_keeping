import tensorflow as tf
import numpy as np
from tensorflow.core.protobuf import saver_pb2
import cnn_model
import data_handler
import data_analyzer
import os
import time
import matplotlib.pyplot as plt


class ModelTrainer:
    def __init__(self, data_handler, epochs = 30, val_split=0.2, L2_norm_const = 0.001, batch_size=100, logs_path='./logs', model_save_path='./save', model_name="model.ckpt"):
        self.epochs = epochs
        self.val_split = val_split
        self.L2_norm_const = L2_norm_const
        self.batch_size = batch_size

        self.logs_path = logs_path
        self.model_save_path = model_save_path
        self.model_name = model_name

        self.val_errors = []

        # initialize saver
        self.saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V1)

        # op to write logs to Tensorboard
        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        self.data_handler = data_handler

        config = tf.ConfigProto()
        config.inter_op_parallelism_threads = 10
        config.intra_op_parallelism_threads = 10

        self.sess = tf.InteractiveSession(config=config)


    def train_model(self):

        train_vars = tf.trainable_variables()
        loss = tf.reduce_mean(tf.square(tf.subtract(cnn_model.y_in, cnn_model.y))) + tf.add_n(
            [tf.nn.l2_loss(v) for v in train_vars]) * self.L2_norm_const
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

        self.sess.run(tf.global_variables_initializer())

        tf.summary.scalar("loss", loss)
        merged_summary_op = tf.summary.merge_all()

        train_data, val_data = self.data_handler.generate_data_splits(self.val_split)

        iterations = []
        loss_values = []

        start_time = time.time()

        for epoch in range(self.epochs):
            print("Epoch " + str(epoch))

            for iteration in range(int(len(train_data) / self.batch_size)):

                train_data_batch_x, train_data_batch_y = self.data_handler.get_train_batch(self.batch_size)

                train_step.run(feed_dict={cnn_model.x: train_data_batch_x,
                                          cnn_model.y_in: np.expand_dims(train_data_batch_y, axis=1),
                                          cnn_model.keep_prob: 0.8})


                if iteration % 10 == 0:
                    val_batch_x, val_batch_y = self.data_handler.get_val_batch(self.batch_size)

                    loss_value = loss.eval(feed_dict={cnn_model.x: val_batch_x,
                                          cnn_model.y_in: np.expand_dims(val_batch_y, axis=1),
                                          cnn_model.keep_prob: 1.0})
                    print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * self.batch_size + iteration, loss_value))

                    loss_values.append(loss_value)
                    iterations.append(epoch * self.batch_size + iteration)


                # write logs at every iteration
                summary = merged_summary_op.eval(feed_dict={cnn_model.x: train_data_batch_x,
                                          cnn_model.y_in: np.expand_dims(train_data_batch_y, axis=1), cnn_model.keep_prob: 1.0})
                self.summary_writer.add_summary(summary, epoch * len(train_data) + iteration)

            self.val_model()
            self.save_model_iteration()

        training_time = time.time() - start_time

        print("Total training time : " + str(training_time))

        self.plot_epoch_errors()

    def plot_epoch_errors(self):
        plt.plot(range(len(self.val_errors)), self.val_errors, 'C1')
        plt.xlabel('Epoch Number')
        plt.ylabel('Validation Error')
        plt.show()
        plt.savefig('validation_error')


    def val_model(self):
        total_loss = 0
        split_iterations = 10

        self.data_handler.val_iterations = 0

        for i in range(split_iterations):
            loss = tf.square(tf.subtract(cnn_model.y_in, cnn_model.y))
            val_batch_x, val_batch_y = self.data_handler.get_val_batch(int(len(self.data_handler.val_data)/split_iterations))

            total_loss += np.mean(loss.eval(feed_dict={cnn_model.x: val_batch_x, cnn_model.y_in: np.expand_dims(val_batch_y, axis=1), cnn_model.keep_prob: 1.0}))

        self.val_errors.append(total_loss/split_iterations)

        print("Error Value after training : " + str(total_loss / split_iterations))

    def save_model_iteration(self):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        checkpoint_path = os.path.join(self.model_save_path, self.model_name)
        filename = self.saver.save(self.sess, checkpoint_path)
        print("Model saved in file: %s" % filename)


if __name__ == '__main__':

    train_simulation = False
    shutdown_on_finish = False
    analyze_data = True

    if train_simulation:
        vec_spec = data_handler.VehicleSpec(angle_norm=1, image_crop_vert=[25,135])
        data_path = './data/augmented_data'
        desc_file = 'augmented_log.csv'
        contains_full_path = True
        model_name = 'sim_model.ckpt'
    else:
        vec_spec = data_handler.VehicleSpec(angle_norm=30, image_crop_vert=[220,480])
        data_path = '/home/elschuer/data/LaneKeepingE2E/train_images_augmented'
        desc_file = 'data_labels.csv'
        contains_full_path = True
        model_name = 'car_model.ckpt'
        convert_image = False
        image_channels = 1

    data_handler = data_handler.DataHandler(data_path, desc_file, vehicle_spec=vec_spec,contains_full_path=contains_full_path, convert_image=convert_image, image_channels=1)

    if analyze_data:
        data_analyzer = data_analyzer.DataAnalyzer()
        data_analyzer.showDataDistribution(data_handler.get_data_y())
        data_analyzer.print_samples_not_equal_zero(data_handler.get_data_y())

    model_trainer = ModelTrainer(epochs=30, data_handler=data_handler, model_name=model_name)
    model_trainer.train_model()

    if shutdown_on_finish:
        os.system("shutdown now -h")

