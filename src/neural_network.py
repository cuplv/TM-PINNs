import jax
import jax.numpy as jnp
from jax.nn import silu
from copy import deepcopy
from .utils import Helpers # Assuming utils.py is a separate file

def feedforward_prediction(params, activations, fn):
    """
    Feedforward neural network model.

    Args:
        params (list): List of network parameters (weights and biases).
        activations (jnp.ndarray): Input activations.
        fn (function): Activation function.

    Returns:
        jnp.ndarray: Network output.
    """
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = fn(outputs)

    w_final, b_final = params[-1]
    final_op = jnp.dot(w_final, activations) + b_final
    return final_op

def batched_prediction(params, activations, fn):
    """
    Batched feedforward neural network prediction.

    Args:
        params (list): List of network parameters (weights and biases).
        activations (jnp.ndarray): Batched input activations.
        fn (function): Activation function.

    Returns:
        jnp.ndarray: Batched network output.
    """
    return jax.vmap(feedforward_prediction, in_axes=(None, 0, None))(params, activations, fn)

def return_nnmodel(hp, ran_key, num_features, num_targets, num_layers, num_neurons_per_layer):
    """
    Initializes and returns a neural network model.

    Args:
        hp (Helpers): An instance of the Helpers class.
        ran_key (jax.random.PRNGKey): JAX random key.
        num_features (int): Number of input features.
        num_targets (int): Number of output targets.
        num_layers (int): Number of hidden layers.
        num_neurons_per_layer (int): Number of neurons per hidden layer.

    Returns:
        list: Initialized network parameters.
    """
    sizes = hp.get_network_layer_sizes(num_features, num_targets, num_layers, num_neurons_per_layer)
    print(sizes)
    net_init_key, ran_key = jax.random.split(ran_key)
    init_params = hp.get_init_network_params(sizes, net_init_key)

    # Run once for JIT compilation
    random_feature_array = jax.random.uniform(ran_key, minval=-1, maxval=1, shape=(num_features, ))
    _ = feedforward_prediction(init_params, random_feature_array, silu)
    params = deepcopy(init_params)

    return params
