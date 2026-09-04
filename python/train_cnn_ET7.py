import numpy as np
import time
from tensorflow import keras
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback

import os
import ast

DATASET_DIR = "../dataset/dataset_ET7/"

with open(os.path.join(DATASET_DIR, "training.txt"), "r") as f:
    contenu = f.read()

debut = contenu.find("{")
fin = contenu.rfind("}")
contenu = contenu[debut:fin+1].replace("{", "[").replace("}", "]")

X_train = np.array(ast.literal_eval(contenu))
print(X_train.shape)  # (100, 48, 13)

Y_int_train = np.array([0]*50 + [1]*50) #création des classes "rouge" = classe 0 et "bleu" = classe 1 (ici Y_int ce n'est que un tableau de 100 cases avec dedans 0 ou 1)
Y_train = to_categorical(Y_int_train, num_classes=2) #donc là, Y il créer un tableau de 100*2 (dans chaque case du tableau de 100 il y'a 2 cases) il regarde si dans Y_int il y'a écrit 0 ou 1 et il met [1,0] si il y'a écrit 0 et [0,1] si il y'a écrit 1


with open(os.path.join(DATASET_DIR, "test.txt"), "r") as f:
    contenu_val = f.read()

debut = contenu_val.find("{")
fin = contenu_val.rfind("}")
contenu_val = contenu_val[debut:fin+1].replace("{", "[").replace("}", "]")

X_val = np.array(ast.literal_eval(contenu_val))
print(X_val.shape) 

Y_int_val = np.array([0]*5 + [1]*5) 
Y_val = to_categorical(Y_int_val, num_classes=2) 

X_train = X_train.reshape(100, 48, 13, 1)
X_val = X_val.reshape(10, 48, 13, 1)


modele = Sequential()
modele.add(Conv2D(8, kernel_size=3, padding='same', activation='relu', input_shape=(48,13,1)))
modele.add(MaxPooling2D(pool_size=(2,2)))     # (24, 6, 8)
modele.add(Dropout(0.2))
modele.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))
modele.add(MaxPooling2D(pool_size=(2,2)))     # (12, 3, 8)
modele.add(Flatten())                          # 288
modele.add(Dropout(0.3))
modele.add(Dense(8, activation='relu'))
modele.add(Dense(2, activation='softmax'))

modele.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
print(modele.summary())

class StopAtValLoss(Callback):  #manière propre de stopper l'entrainement quand la MSE  de validation à dépassé 0.05
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        if val_loss is not None and val_loss < self.threshold:
            print(f"\n>>> val_MSE = {val_loss:.4f} < {self.threshold} a l'epoch {epoch+1}. Arret.")
            self.model.stop_training = True

stopper = StopAtValLoss(threshold=0.05)

t0 = time.time()

history = modele.fit(X_train, Y_train,
                     epochs=500,                     
                     batch_size=32,
                     validation_data=(X_val, Y_val),
                     callbacks=[stopper])

duree = time.time() - t0

print(f"Duree               : {duree:.2f} s")
print(f"Nombre d'epochs     : {len(history.history['loss'])}")
print(f"MSE finale (train)  : {history.history['loss'][-1]:.4f}")
print(f"MSE finale (test)   : {history.history['val_loss'][-1]:.4f}")
print(f"Accuracy train      : {history.history['accuracy'][-1]:.4f}")
print(f"Accuracy test       : {history.history['val_accuracy'][-1]:.4f}")


# Prends le tout premier echantillon de training (qui est un "rouge")
# sample = X_train[0].reshape(48, 13)

# export_arduino_header(modele, "model_weights.h")