def VGG_16():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', input_shape=(64, 64, 3), data_format='channels_last'))
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', data_format='channels_last'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(128, (3, 3), padding='valid', strides=(1, 1), activation='relu', data_format='channels_last'))
    model.add(Conv2D(128, (3, 3), padding='valid', strides=(1, 1), activation='relu', data_format='channels_last'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(256, (3, 3), padding='valid', strides=(1, 1), activation='relu', data_format='channels_last'))
    model.add(Conv2D(256, (3, 3), padding='valid', strides=(1, 1), activation='relu', data_format='channels_last'))
    model.add(Conv2D(256, (3, 3), padding='valid', strides=(1, 1), activation='relu', data_format='channels_last'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='softmax'))

    return model