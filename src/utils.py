import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os

class Helpers:
    """
    Helper class for network initialization and plotting.
    """
    def __init__(self):
        pass

    def get_network_layer_sizes(self, num_features, num_targets, num_layers, num_neurons_per_layer):
        """
        Determines the layer sizes for a neural network.

        Args:
            num_features (int): Number of input features.
            num_targets (int): Number of output targets.
            num_layers (int): Number of hidden layers.
            num_neurons_per_layer (int): Number of neurons per hidden layer.

        Returns:
            list: A list of layer sizes.
        """
        sizes = [num_features] + [num_neurons_per_layer] * num_layers + [num_targets]
        return sizes

    def get_init_network_params(self, sizes, key):
        """
        Initializes the network parameters (weights and biases).

        Args:
            sizes (list): A list of layer sizes.
            key (jax.random.PRNGKey): JAX random key.

        Returns:
            list: A list of initialized network parameters.
        """
        params = []
        for i in range(len(sizes) - 1):
            key, w_key, b_key = jax.random.split(key, 3)
            weight_shape = (sizes[i+1], sizes[i])
            bias_shape = (sizes[i+1], 1)

            # Xavier initialization
            limit = jnp.sqrt(6 / (sizes[i] + sizes[i+1]))
            w = jax.random.uniform(w_key, shape=weight_shape, minval=-limit, maxval=limit)
            b = jax.random.uniform(b_key, shape=bias_shape, minval=-limit, maxval=limit)[:, 0] # Flatten bias to 1D

            params.append((w, b))
        return params

    def plot_creations_2d(self, t_eval, test_dataset, test_target, steps, name, no_of_params, no_of_svariables, no_plot_figures, set, pred_batch=None):
        """
        Plots the predicted and target trajectories.

        Args:
            t_eval (jnp.ndarray): Array of time points.
            test_dataset (jnp.ndarray): Test dataset.
            test_target (jnp.ndarray): Target data.
            steps (int): Number of steps per trajectory.
            name (str): Name for saving the plot.
            no_of_params (int): Number of parameters in the system.
            no_of_svariables (int): Number of state variables in the system.
            no_plot_figures (int): Number of figures to plot.
            set (str): Dataset name (e.g., 'train', 'test').
            pred_batch (jnp.ndarray, optional): Predicted data batch. Defaults to None.
        """
        os.makedirs(f"figs/{set}", exist_ok=True)
        for i in range(no_plot_figures):
            plt.figure(figsize=(10, 6))
            start_idx = i * steps
            end_idx = start_idx + steps
            for j in range(no_of_svariables):
                plt.plot(t_eval, test_target[start_idx:end_idx, j], label=f'Target State {j+1}', linestyle='--')
                if pred_batch is not None:
                    plt.plot(t_eval, pred_batch[start_idx:end_idx, j], label=f'Prediction State {j+1}')
            plt.xlabel('Time')
            plt.ylabel('State Variable Value')
            plt.title(f'Trajectory {i+1}')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"figs/{set}/{name}_trajectory_{i+1}.png")
            plt.close()


    def plot_error_distributions(self, pred_batch, test_target, set, name, no_of_svariables):
        """
        Plots the distribution of absolute errors.

        Args:
            pred_batch (jnp.ndarray): Predicted data batch.
            test_target (jnp.ndarray): Target data.
            set (str): Dataset name (e.g., 'train', 'test').
            name (str): Name for saving the plot.
            no_of_svariables (int): Number of state variables in the system.
        """
        os.makedirs(f"figs/{set}", exist_ok=True)
        absolute_error = jnp.abs(pred_batch - test_target)
        for j in range(no_of_svariables):
            plt.figure(figsize=(10, 6))
            plt.hist(absolute_error[:, j], bins=50)
            plt.xlabel(f'Absolute Error for State {j+1}')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of Absolute Errors for State {j+1}')
            plt.grid(True)
            plt.savefig(f"figs/{set}/{name}_error_distribution_state_{j+1}.png")
            plt.close()