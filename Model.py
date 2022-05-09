import keras as keras
import numpy as np

import pandas as pd

from pathlib import Path

import os.path
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt

from tensorflow import keras
from keras.models import save_model , load_model
import scipy

from keras.preprocessing.image import load_img, img_to_array

# Create a List with the filepaths for training and testing 3
train_dir = Path('train')
train_filepaths = list(train_dir.glob(r'**/*.jpg'))

test_dir = Path('test')
test_filepaths = list(test_dir.glob(r'**/*.jpg'))

val_dir = Path("validation")
val_filepaths = list(test_dir.glob(r'**/*.jpg'))


def image_processing(filepath):
    labels = [str(filepath[i]).split("\\")[-2] \
              for i in range(len(filepath))]

    filepath = pd.Series(filepath, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')
    # Concatenate filepaths and Labels
    df = pd.concat([filepath, labels], axis=1)
    # Shuffle the DataFrame and reset index
    df = df.sample(frac=1).reset_index(drop=True)
    return df


train_df = image_processing(train_filepaths)
test_df = image_processing(test_filepaths)
val_df = image_processing(val_filepaths)

print('-- Training set --\n')
print(f'Number of pictures: {train_df.shape[0]}\n')
print(f'Number of different labels: {len(train_df.Label.unique())}\n')
print(f'Labels: {train_df.Label.unique()}')

train_df.head(5)

# Create a DataFrame with one Label of each category

df_unique = train_df.copy().drop_duplicates(subset=["Label"]).reset_index()

# Display some pictures of the dataset

fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(8, 7),

                         subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(df_unique.Filepath[i]))
    ax.set_title(df_unique.Label[i], fontsize=12)

plt.tight_layout(pad=0.5)

plt.show()


# Model Building
train_generator = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=keras.applications.mobilenet_v2.preprocess_input

)

test_generator = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=keras.applications.mobilenet_v2.preprocess_input

)

train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col="Filepath",
    y_col="Label",
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=0,
    rotation_range=30,
    zoom_range=06.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"

)

val_images = train_generator.flow_from_dataframe(
    dataframe=val_df,
    x_col="Filepath",
    y_col="Label",
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=0,
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"

)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col="Filepath",
    y_col="Label",
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    shuffle=False
)

pretrained_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg"
)
pretrained_model.trainable = False

inputs = pretrained_model.input
x = keras.layers.Dense(128, activation='relu')(pretrained_model.output)
x = keras.layers.Dense(128, activation='relu')(x)

outputs = keras.layers.Dense(36, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss="categorical_crossentropy",
    metrics=["accuracy"]

)

history = model.fit(
    train_images,
    validation_data=val_images,
    batch_size=32,
    epochs=5,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]

)

# Predict the Label of the test images

pred = model.predict(test_images)

pred = np.argmax(pred, axis=1)

# Map the Label

labels = (train_images.class_indices)
labels = dict((v, k) for k, v in labels.items())
predl = [labels[k] for k in pred]


def output(location):
    img = load_img(location, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    return res

print()

model.save("model.h5")

