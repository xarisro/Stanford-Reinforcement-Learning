import tensorflow as tf

def build_mlp(
        mlp_input,
        output_size,
        scope,
        n_layers,
        size,
        output_activation=None):
    """
    Build a feed forward network (multi-layer perceptron, or mlp)
    with 'n_layers' hidden layers, each of size 'size' units.
    Use tf.nn.relu nonlinearity between layers.
    Args:
          mlp_input: the input to the multi-layer perceptron
          output_size: the output layer size
          scope: the scope of the neural network
          n_layers: the number of hidden layers of the network
          size: the size of each layer:
          output_activation: the activation of output layer
    Returns:
          The tensor output of the network

    TODO: Implement this function. This will be similar to the linear
    model you implemented for Assignment 2.
    "tf.layers.dense" and "tf.variable_scope" may be helpful.

    A network with n hidden layers has n 'linear transform + nonlinearity'
    operations followed by the final linear transform for the output layer
    (followed by the output activation, if it is not None).

    """
    #######################################################
    #########   YOUR CODE HERE - 7-20 lines.   ############

    with tf.variable_scope(scope):
        prev_layer = mlp_input
        
        for i in range(n_layers):
            next_layer = tf.contrib.layers.fully_connected(inputs=prev_layer, 
                                                           num_outputs=size, 
                                                           activation_fn=tf.nn.relu)
            prev_layer = next_layer

        out = tf.contrib.layers.fully_connected(inputs=prev_layer, 
                                                num_outputs=output_size, 
                                                activation_fn=output_activation)

    return out
    #######################################################
    #########          END YOUR CODE.          ############