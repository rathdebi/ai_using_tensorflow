import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from constants import PATH
from pathlib import Path

# set train human and horse directory
train_human_dir = Path(PATH).joinpath("humans")
train_horses_dir = Path(PATH).joinpath("horses")
# verify all file names in train human and horses folder
train_human_file_names = [entry.name for entry in train_human_dir.iterdir() if entry.is_file()]
train_horses_file_names = [entry.name for entry in train_horses_dir.iterdir() if entry.is_file()]

print("training human images:::==>",len(train_human_file_names))
print("training horse images:::==>", len(train_horses_file_names))

model = Sequential([Conv2D(16, (3,3), activation="relu",input_shape=(300,300,3)),
                    MaxPooling2D(2,2),
                    Conv2D(32, (3,3),activation="relu"),
                    MaxPooling2D(2,2),
                    Conv2D(64, (3,3), activation="relu"),
                    MaxPooling2D(2,2),
                    Conv2D(64, (3,3), activation="relu"),
                    MaxPooling2D(2,2),
                    Conv2D(64, (3,3), activation="relu"),
                    MaxPooling2D(2,2),
                    Flatten(),
                    Dense(512, activation="relu"),
                    Dense(1, activation="sigmoid")
                ]
            )
model.summary()
# compile the model
model.compile(loss="binary_crossentropy",
              optimizer=RMSprop(learning_rate=0.001),
              metrics=["accuracy"]
              )

# create a image data generator
train_data_gen= ImageDataGenerator(rescale=1./255)
train_data_generator = train_data_gen.flow_from_directory(
    Path(PATH),
    target_size=(300,300),
    batch_size=128,
    class_mode="binary",
)
history = model.fit(
    train_data_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=1,
)





# # validation_data_gen= ImageDataGenerator(rescale=1./255)
# # vaildation_data_generator = validation_data_gen.flow_from_directory(
# #     validation_dir,
# #     target_size=(300,300),
# #     batch_size=128,
# #     class_mode="binary",

# # )

