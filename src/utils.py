import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

class Helpers:
    """
    Helper class for network initialization and plotting.
    """
    def __init__(self):
        pass

    def get_random_layer_params(self, m, n, ran_key, scale=0.01):
        """Helper function to randomly initialize weights and biases using the JAX-defined randoms."""
        w_key, b_key = jax.random.split(ran_key)
        ran_weights = scale * jax.random.normal(w_key, (n, m))
        ran_biases = scale * jax.random.normal(b_key, (n,)) 
        return ran_weights, ran_biases


    def get_xavier_init_layer_params(self, m, n, ran_key):
        """Helper function to initialize weights and biases using the Xavier initialization."""
        w_key, b_key = jax.random.split(ran_key)
        glorot_scale = np.sqrt(2.0 / (m + n))
        ran_weights = glorot_scale * jax.random.normal(w_key, (n, m))
        ran_biases = glorot_scale * jax.random.normal(b_key, (n,))
        return ran_weights, ran_biases


    def get_init_network_params(self, sizes, ran_key):
        """Initialize all layers for a fully-connected neural network."""
        keys = jax.random.split(ran_key, len(sizes))
        return [self.get_xavier_init_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
        # return [self.get_random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


    def get_network_layer_sizes(self, n_features, n_targets, n_layers, n_neurons_per_layer):
        """Helper function to get the sizes of the layers of a feedforward neural network."""
        dense_layer_sizes = [n_neurons_per_layer]*n_layers
        layer_sizes = [n_features, *dense_layer_sizes, n_targets]
        return layer_sizes


    def plot_creations_2d(self, t_eval, dataset, target, steps, name, no_of_params, no_of_svariables, state_vars_symbols, no_plot_figures, set, pred_batch=None):
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
        dataset = dataset[:no_plot_figures*steps, :]
        pred_batch = pred_batch[:no_plot_figures*steps, :]
        target = target[:no_plot_figures*steps, :]
        
        fig, axs = plt.subplots(2, 5, figsize=(25, 10))
        axs = axs.flatten()
        markers = ['p', 'x', 'v', 'o', '1', '2', '3', '4', '8', 's', '<', '>', 'D', 'd', '|']

        for i in range(no_plot_figures):
            ax = axs[i]

            for ind, svar in enumerate(range(no_of_svariables)):
                ax.plot(t_eval, np.array(target[(i*steps):(i*steps)+steps, svar]), color='green', marker=markers[svar], label=fr"$gt_{state_vars_symbols[svar]}$")
            
            if "test" in set.lower():
                for ind, testsvar in enumerate(range(no_of_svariables)):
                    ax.plot(t_eval, np.array(pred_batch[(i*steps):(i*steps)+steps, testsvar]), color='blue', marker=markers[testsvar], label=fr"$pred_{state_vars_symbols[testsvar]}$")

            if no_of_svariables <= 6:
                ax.legend()
            ax.grid()
            ax.set_xlabel(r"$t$")

            ylabel_str = ""
            for ind, svar in enumerate(range(no_of_svariables-1)):
                ylabel_str += fr"${state_vars_symbols[svar]}(t), $"
            ylabel_str += fr"${state_vars_symbols[no_of_svariables-1]}(t)$"
            ax.set_ylabel(ylabel_str)

            params = dataset[(i*steps), :]
            initcond_str = ", ".join([f"{p:.2f}" for p in params[1:no_of_svariables]])
            param_str = ", ".join([f"{p:.2f}" for p in params[no_of_svariables:no_of_svariables+no_of_params]])
            ax.set_title(fr"$init={initcond_str}$" + "\n" + fr"$params={param_str}$", multialignment='center')

        plt.tight_layout()
        plt.savefig(f'figs/{set}/{name}.png', dpi=300)
        plt.close()

    def plot_error_distributions(self, preds, gt, set, name, no_of_svariables):
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
        
        diff = abs(preds - gt)

        _, axs = plt.subplots(1, no_of_svariables, figsize=(24, 5))
        for i in range(no_of_svariables):
            sns.histplot(diff[:, i], ax=axs[i], alpha=0.5, bins=30, kde=True, fill=True, color="green", label=r"$x_{\text{error}}$")
            axs[i].axvline(diff[:, i].mean(), color='blue', linestyle='--', label=rf"$\mu={diff[:, i].mean():.2f}$")
            axs[i].legend()
            axs[i].set_title(r"Absolute Error distribution in $X_{error}$")
            axs[i].set_xlabel("Error")
            axs[i].grid()
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"figs/{set}/error_distribution_{name}.png", dpi=300)
        plt.close()
