import tensorflow as tf


def build_mlp_architecture_with_two_inputs(
    input_1_size, input_2_size, width, depth, activation, output_size, name
):
    input_1 = tf.keras.Input(shape=input_1_size)
    input_2 = tf.keras.Input(shape=input_2_size)

    input_combined = tf.keras.layers.Concatenate()([input_1, input_2])

    layers = [
        tf.keras.layers.Dense(units=width, activation=activation) for _ in range(depth)
    ]
    xx = input_combined
    for layer in layers:
        xx = layer(xx)

    xx = tf.keras.layers.Dense(units=output_size)(xx)

    return tf.keras.Model(inputs=[input_1, input_2], outputs=xx, name=name)
