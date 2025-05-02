import jax
import jax.numpy as jnp
from jax.nn import silu
from copy import deepcopy

def feedforward_prediction(params, activations, fn):
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = fn(outputs)

    w_final, b_final = params[-1]
    final_op = jnp.dot(w_final, activations) + b_final
    return final_op

def return_nnmodel(hp, ran_key, num_features, num_targets, num_layers, num_neurons_per_layer):
    sizes = hp.get_network_layer_sizes(num_features, num_targets, num_layers, num_neurons_per_layer)
    print(sizes)
    net_init_key, ran_key = jax.random.split(ran_key)
    init_params = hp.get_init_network_params(sizes, net_init_key)

    # Run once for JIT compilation
    random_feature_array = jax.random.uniform(ran_key, minval=-1, maxval=1, shape=(num_features, ))
    _ = feedforward_prediction(init_params, random_feature_array, silu)
    params = deepcopy(init_params)

    return params
