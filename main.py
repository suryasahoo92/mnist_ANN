
import tensorflow as tf
import numpy as np
from utils.all_utils import prepare_data,save_model,save_test
from utils.model import classification_model
import logging
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str, filemode="a")


LAYERS = [
          tf.keras.layers.Flatten(input_shape=[28,28], name="inputLayer"),
          tf.keras.layers.Dense(300, activation="relu", name = "hiddenLayer1"),
          tf.keras.layers.Dense(100, activation="relu", name = "hiddenLayer2"),
          tf.keras.layers.Dense(10, activation ="softmax", name = "outputLayer")
    ]

LOSS_FUNCTION = "sparse_categorical_crossentropy"
OPTIMIZER = "SGD"
METRICS = ["accuracy"]

df = tf.keras.datasets.mnist
X_train,y_train,X_valid, y_valid, X_test,y_test = prepare_data(df)

EPOCHS = 30
VALIDATION = (X_valid, y_valid)

try:
    #model = Model(layers=LAYERS,loss_function=LOSS_FUNCTION,optimizer=OPTIMIZER,metrics=METRICS)
    history = classification_model(layers=LAYERS,loss_function=LOSS_FUNCTION,optimizer=OPTIMIZER,metrics=METRICS,X_v=X_train,y_v=y_train,epochs=EPOCHS,validation_data=VALIDATION)

except Exception as e:
    logging.exception(e)
    raise e

"""
try:
    if __name__ == '__main__':
        logging.info(">>>>> starting training >>>>>")
        history = model.fit_data(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION)
        logging.info("<<<<< training done successfully<<<<<\n")

        logging.info("Saving the model")
        save_model(history, filename="trained_model.model")

except Exception as e:
    logging.exception(e)
    raise e
"""
#history.evaluate(X_test,y_test)
save_model(history, filename="trained_model.model")
y_prob = history.predict(X_test)

y_pred = np.argmax(y_prob, axis=-1)

save_test(y_test,y_pred)




