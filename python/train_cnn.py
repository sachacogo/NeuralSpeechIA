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


import ast

DATASET_DIR = "../dataset/dataset_perso/"

def export_arduino_header(model, filename="model_weights.h"):
    weighted_layers = [l for l in model.layers if len(l.get_weights()) > 0]
    c_names = ["conv1", "conv2", "dense1", "dense2"]

    # --- derivation automatique des dimensions depuis le modele ---
    conv_layers  = [l for l in model.layers if isinstance(l, Conv2D)]
    pool_layers  = [l for l in model.layers if isinstance(l, MaxPooling2D)]
    dense_layers = [l for l in model.layers if isinstance(l, Dense)]

    input_h, input_w, input_c = model.input_shape[1:]
    conv1_filters = conv_layers[0].filters
    conv2_filters = conv_layers[1].filters
    kernel_size   = conv_layers[0].kernel_size[0]

    # same padding => Conv2D ne change pas H/W. MaxPool 2x2 divise par 2 (floor).
    pool1_h = input_h // pool_layers[0].pool_size[0]
    pool1_w = input_w // pool_layers[0].pool_size[1]
    pool2_h = pool1_h // pool_layers[1].pool_size[0]
    pool2_w = pool1_w // pool_layers[1].pool_size[1]
    flat_size = pool2_h * pool2_w * conv2_filters

    dense1_units = dense_layers[0].units
    num_classes  = dense_layers[1].units

    def array_to_c(arr, indent="  "):
        flat = arr.flatten()
        lines = []
        for i in range(0, len(flat), 8):
            chunk = flat[i:i+8]
            lines.append(indent + ", ".join(f"{v:+.8f}f" for v in chunk))
        return ",\n".join(lines)

    with open(filename, "w", encoding="utf-8") as f:
        f.write("#pragma once\n\n")
        f.write("#define _1_OPTIMIZE B00010000\n\n")
        f.write("#define Sigmoid\n\n")
        f.write("#define NumberOf(arg) ((unsigned int) (sizeof (arg) / sizeof (arg [0])))\n\n")

        f.write("// Dimensions du reseau (derivees du modele Keras)\n")
        f.write(f"#define INPUT_H        {input_h:>4}\n")
        f.write(f"#define INPUT_W        {input_w:>4}\n")
        f.write(f"#define INPUT_C        {input_c:>4}\n")
        f.write(f"#define CONV1_FILTERS  {conv1_filters:>4}\n")
        f.write(f"#define CONV2_FILTERS  {conv2_filters:>4}\n")
        f.write(f"#define KERNEL_SIZE    {kernel_size:>4}\n")
        f.write(f"#define POOL_H         {pool1_h:>4}\n")
        f.write(f"#define POOL_W         {pool1_w:>4}\n")
        f.write(f"#define POOL2_H        {pool2_h:>4}\n")
        f.write(f"#define POOL2_W        {pool2_w:>4}\n")
        f.write(f"#define FLAT_SIZE      {flat_size:>4}\n")
        f.write(f"#define DENSE1_UNITS   {dense1_units:>4}\n")
        f.write(f"#define NUM_CLASSES    {num_classes:>4}\n\n")

        for layer, cname in zip(weighted_layers, c_names):
            w, b = layer.get_weights()
            f.write(f"// ----- {layer.name} -----\n")
            f.write(f"// poids : shape {w.shape}  ({w.size} valeurs)\n")
            f.write(f"PROGMEM const float {cname}_w[{w.size}] = {{\n")
            f.write(array_to_c(w))
            f.write("\n};\n\n")

            f.write(f"// biais : shape {b.shape}  ({b.size} valeurs)\n")
            f.write(f"PROGMEM const float {cname}_b[{b.size}] = {{\n")
            f.write(array_to_c(b))
            f.write("\n};\n\n")

        total = sum(l.count_params() for l in weighted_layers)
        f.write(f"// Total parametres : {total}\n")
        f.write(f"// Taille flash (float32) : ~{total * 4 / 1024:.1f} KB\n")

    print(f"Fichier '{filename}' genere ({total} parametres, ~{total*4/1024:.1f} KB).")

def charger(fichier):
    with open(fichier, "r") as f:
        contenu = f.read()
    debut = contenu.find("{")
    fin   = contenu.rfind("}")
    contenu = contenu[debut:fin+1].replace("{", "[").replace("}", "]")
    return np.array(ast.literal_eval(contenu))

X_bonjour  = charger(DATASET_DIR + "bonjour.txt")
X_aurevoir = charger(DATASET_DIR + "merci.txt")
X_val      = charger(DATASET_DIR + "datatest.txt")
X_train = np.concatenate([X_bonjour, X_aurevoir], axis=0)

Y_int_train = np.array([0]*len(X_bonjour) + [1]*len(X_aurevoir))
Y_train = to_categorical(Y_int_train, num_classes=2)

Y_int_val = np.array([0]*5 + [1]*5)
Y_val = to_categorical(Y_int_val, num_classes=2)

def augmenter(X):
    outs = [X]
    for s in (-3, -1, 1, 3):
        outs.append(np.roll(X, s, axis=1))
    return np.concatenate(outs, 0)
X_train = augmenter(X_train)
Y_train = np.tile(Y_train, (5, 1))

X_train = X_train.reshape(-1, 48, 13, 1)
X_val   = X_val.reshape(-1, 48, 13, 1)

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

# class StopAtValLoss(Callback):  #manière propre de stopper l'entrainement quand la MSE  de validation à dépassé 0.05
#     def __init__(self, threshold):
#         super().__init__()
#         self.threshold = threshold
#     def on_epoch_end(self, epoch, logs=None):
#         val_loss = logs.get('val_loss')
#         if val_loss is not None and val_loss < self.threshold:
#             print(f"\n>>> val_MSE = {val_loss:.4f} < {self.threshold} a l'epoch {epoch+1}. Arret.")
#             self.model.stop_training = True

# stopper = StopAtValLoss(threshold=0.01)

t0 = time.time()

history = modele.fit(X_train, Y_train,
                     epochs=30,                     
                     batch_size=32,
                     validation_data=(X_val, Y_val),
                    #  callbacks=[stopper]
                     )


duree = time.time() - t0

print(f"Duree               : {duree:.2f} s")
print(f"Nombre d'epochs     : {len(history.history['loss'])}")
print(f"MSE finale (train)  : {history.history['loss'][-1]:.4f}")
print(f"MSE finale (test)   : {history.history['val_loss'][-1]:.4f}")
print(f"Accuracy train      : {history.history['accuracy'][-1]:.4f}")
print(f"Accuracy test       : {history.history['val_accuracy'][-1]:.4f}")


# Prends le tout premier echantillon de training (qui est un "rouge")
# sample = X_train[0].reshape(48, 13)

export_arduino_header(modele, "model_weights.h")

