import tensorflow as tf

def make_model(learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=lr
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model