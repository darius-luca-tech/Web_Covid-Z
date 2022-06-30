
# import pachetele necesare
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from imutils import paths
from sklearn.metrics import classification_report
from pyimagesearch import config
from pyimagesearch.resnet import ResNet
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use("Agg")


# setări pentru a antrena rețeaua folosind placa video
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


# setez parser-ul pentru argumentele script-ului
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to trained model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# definesc numărul total de epoci pentru antrenare, rata de învățare inițială și dimensiunea lotului
NUM_EPOCHS = 1500
INIT_LR = 1e-1
BS = 64


def poly_decay(epoch):
    # inițializez numărul maxim de epoci, rata de învățare inițială și puterea polinomului
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0

    # calculez noua rată de învățare pe baza polynomial decay
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

    # returnez noua rată de învățare
    return alpha


# determin numărul total de imagini din directoarele de antrenare, validare și testare
totalTrain = len(list(paths.list_images("dataset/training")))
totalVal = len(list(paths.list_images("dataset/validation")))
totalTest = len(list(paths.list_images("dataset/testing")))

# inițializez obiectul pentru augmentarea datelor de antrenament
trainAug = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=20,
    zoom_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest")

# inițializez obiectul pentru augmentarea datelor de validare și testare
valAug = ImageDataGenerator(rescale=1 / 255.0)

# inițializez generatorul pentru antrenare
trainGen = trainAug.flow_from_directory(
    "dataset/training",
    class_mode="categorical",
    target_size=(64, 64),
    color_mode="rgb",
    shuffle=True,
    batch_size=BS)

# inițializez generatorul pentru validare
valGen = valAug.flow_from_directory(
    "dataset/validation",
    class_mode="categorical",
    target_size=(64, 64),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS)

# inițializez generatorul pentru testare
testGen = valAug.flow_from_directory(
    "dataset/testing",
    class_mode="categorical",
    target_size=(64, 64),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS)

# inițializez modelul ResNet și îl compilez
model = ResNet.build(64, 64, 3, 2, (3, 4, 6),
                     (64, 128, 256, 512), reg=0.0005)
opt = SGD(lr=INIT_LR, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# definesc setul de callback-uri și apelez funcția de antrenare a modelului
callbacks = [LearningRateScheduler(poly_decay)]
H = model.fit_generator(
    trainGen,
    steps_per_epoch=totalTrain // BS,
    validation_data=valGen,
    validation_steps=totalVal // BS,
    epochs=NUM_EPOCHS,
    callbacks=callbacks)

# resetez generatorul de testare și apoi utilizez modelul antrenat pentru a face predicții asupra datelor
print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict_generator(testGen,
                                   steps=(totalTest // BS) + 1)

# pentru fiecare imagine din setul de testare trebuie să găsim indexul etichetei cu cea mai mare probabilitate prezisă
predIdxs = np.argmax(predIdxs, axis=1)

# afișez un raport de clasificare frumos formatat
print(classification_report(testGen.classes, predIdxs,
                            target_names=testGen.class_indices.keys()))


# salvez modelul antrenat pe disk
print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"])

# afișez un grafic cu loss-ul și acuratețea
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")

plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
