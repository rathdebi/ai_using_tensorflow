import tensorflow as tf

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch:int, logs:dict={}):
        if logs.get("loss") < 0.40:
            print("\n loss less than .40, hence cancelling training")
            self.model.stop_training = True
        