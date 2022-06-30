# import pachetele necesare
from pyimagesearch import config
from imutils import paths
import random
import shutil
import os

# iau path-urile către toate imaginile de intrare din directorul de intrare original și le amestec
imagePaths = list(paths.list_images(config.ORIG_INPUT_DATASET))
random.seed(42)
random.shuffle(imagePaths)

# calculez împărțirea datelor pentru antrenamentului și testare
i = int(len(imagePaths) * config.TRAIN_SPLIT)
trainPaths = imagePaths[:i]
testPaths = imagePaths[i:]

# voi folosi o parte din datele de antrenare pentru a realiza validarea
i = int(len(trainPaths) * config.VAL_SPLIT)
valPaths = trainPaths[:i]
trainPaths = trainPaths[i:]

# definesc seturile de date pe care le voi construi
datasets = [
    ("training", trainPaths, config.TRAIN_PATH),
    ("validation", valPaths, config.VAL_PATH),
    ("testing", testPaths, config.TEST_PATH)
]

# parcurg seturile de date
for (dType, imagePaths, baseOutput) in datasets:
    # afisez ce set de date se creează
    print("[INFO] building '{}' split".format(dType))

    # dacă directorul de ieșire nu există, se creează
    if not os.path.exists(baseOutput):
        print("[INFO] 'creating {}' directory".format(baseOutput))
        os.makedirs(baseOutput)

        # parcurg path-urile imaginilor de intrare
    for inputPath in imagePaths:
        # extrage numele fișierului imaginii de intrare împreună cu eticheta clasei corespunzătoare
        filename = inputPath.split(os.path.sep)[-1]
        label = inputPath.split(os.path.sep)[-2]

        # construiesc calea către directorul cu etichete
        labelPath = os.path.sep.join([baseOutput, label])

        # dacă directorul de ieșire pentru etichete nu există, se creează
        if not os.path.exists(labelPath):
            print("[INFO] 'creating {}' directory".format(labelPath))
            os.makedirs(labelPath)

        # construiesc calea către imaginea de destinație și apoi copiez imaginea
        p = os.path.sep.join([labelPath, filename])
        shutil.copy2(inputPath, p)
