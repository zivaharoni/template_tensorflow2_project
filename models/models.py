from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Conv2D, Softmax

def build_model(config):
    if config.model_name == "mnist-cnn":
        model = {'train': CNNModel(config),
                 'eval':  CNNModel(config) }

    else:
        raise ValueError("'{}' is an invalid model name")

    return model

def CNNModel(config):

    model = Sequential()
    model.add(Conv2D(config.hidden_size[0], (3, 3), activation='relu', input_shape=config.model_input_dim))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(config.hidden_size[1], (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(config.model_output_dim, activation=None))
    model.add(Softmax(axis=-1))

    return model
