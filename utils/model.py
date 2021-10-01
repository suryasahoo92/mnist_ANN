import tensorflow as tf
import logging
import os



# class Model:
#     def __init__(self,layers,loss_function,optimizer,metrics):
#         self.layers = layers
#         self.loss_function = loss_function
#         self.optimizer = optimizer
#         self.metrics = metrics
#         # self.X = X
#         # self.y = y
#         # self.epochs = epochs
#         # self.validation_data = validation_data
        
def classification_model(layers,loss_function,optimizer,metrics,X_v,y_v,epochs,validation_data):
    model_clf = tf.keras.models.Sequential(layers)
    model_clf.summary()
    print("summary completed")
    model_clf.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
    print("I am before fit")
    model_clf.fit(X_train=X_v,y_train=y_v,epochs=epochs,validation_data=validation_data)
    print("I am after fit")
    return model_clf