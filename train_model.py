import numpy as np
from keras.optimizers import Adam
import cnn_model
import data_handler
import data_analyzer
import vehicle_spec
import os
import time
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, data_handler, epochs = 30, val_split=0.2, batch_size=100, logs_path='./logs', model_save_path='./save', model_name="model.h5"):
        self.epochs = epochs
        self.val_split = val_split
        self.batch_size = batch_size

        self.logs_path = logs_path
        self.model_save_path = model_save_path
        self.model_name = model_name

        self.data_handler = data_handler

    def train_model(self):

        model = cnn_model.get_model()
        model.compile(optimizer=Adam(lr=1e-4), loss='mse')
        model.summary()

        images = np.array(data_handler.x_data)
        angles = data_handler.y_data

        print("Images size", images.shape)

        start_time = time.time()

        history = model.fit(x=images, y=angles, batch_size=100, epochs=self.epochs, validation_split=self.val_split, shuffle=True)
        model.save(self.model_save_path+"/"+self.model_name)

        training_time = time.time() - start_time
        print("Total training time : " + str(training_time))

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, color='red', label='Training loss')
        plt.plot(epochs, val_loss, color='green', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


if __name__ == '__main__':

    train_simulation = False
    shutdown_on_finish = False
    analyze_data = False

    if train_simulation:
        vec_spec = vehicle_spec.VehicleSpec(angle_norm=1, image_crop_vert=[25,135])
        data_path = './data/augmented_data'
        desc_file = 'augmented_log.csv'
        contains_full_path = True
        model_name = 'sim_model.ckpt'
    else:
        vec_spec = vehicle_spec.VehicleSpec(angle_norm=30, image_crop_vert=[220,480])
        data_path = '/home/elschuer/data/LaneKeepingE2E/images_train_augmented/'
        desc_file = 'data_labels.csv'
        contains_full_path = True
        model_name = 'nvidia_model.h5'
        convert_image = False
        image_channels = 1

    data_handler = data_handler.DataHandler(data_path, desc_file, vehicle_spec=vec_spec,contains_full_path=contains_full_path, convert_image=convert_image, image_channels=1)
    data_handler.read_data()

    if analyze_data:
        data_analyzer = data_analyzer.DataAnalyzer()
        data_analyzer.showDataDistribution(data_handler.y_data)
        data_analyzer.print_samples_not_equal_zero(data_handler.y_data)

    model_trainer = ModelTrainer(epochs=10, data_handler=data_handler, model_name=model_name)
    model_trainer.train_model()

    if shutdown_on_finish:
        os.system("shutdown now -h")

