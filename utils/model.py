import tensorflow as tf



class Model:
    def __init__(self,layers,epoch,validation,loss_function,optimizer,metrics):
        self.layers = layers
        self.epoch = epoch
        self.validation = validation
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.metrics = metrics

    def classification_model(self):
        model_clf = tf.keras.models.Sequential(self.layers)
        model_clf.summary()
        model_clf.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics)
