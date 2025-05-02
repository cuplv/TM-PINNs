import jax
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class Helpers:
    def __init__(self) -> None:
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


    def plot_creations_2d(self, t_eval, dataset, target, steps, name, no_of_params, no_of_svariables, no_plot_figures=10, set: str="train", pred_batch=None):
        """Function to plot the target dataset and the original dataset"""
        dataset = dataset[:no_plot_figures*steps, :]
        pred_batch = pred_batch[:no_plot_figures*steps, :]
        target = target[:no_plot_figures*steps, :]
        
        fig, axs = plt.subplots(2, 5, figsize=(25, 10))
        axs = axs.flatten()

        for i in range(no_plot_figures):
            ax = axs[i]

            if no_of_svariables == 2:
                ax.plot(t_eval, np.array(target[(i*steps):(i*steps)+steps, 0]), color='blue', marker='p', label=r"$gt_x$")
                ax.plot(t_eval, np.array(target[(i*steps):(i*steps)+steps, 1]), color='blue', marker='x', label=r"$gt_y$")
            elif no_of_svariables == 3:
                ax.plot(t_eval, np.array(target[(i*steps):(i*steps)+steps, 0]), color='blue', marker='p', label=r"$gt_x$")
                ax.plot(t_eval, np.array(target[(i*steps):(i*steps)+steps, 1]), color='blue', marker='x', label=r"$gt_y$")
                ax.plot(t_eval, np.array(target[(i*steps):(i*steps)+steps, 2]), color='blue', marker='v', label=r"$gt_z$")
            elif no_of_svariables == 4:
                ax.plot(t_eval, np.array(target[(i*steps):(i*steps)+steps, 0]), color='blue', marker='p', label=r"$gt_x$")
                ax.plot(t_eval, np.array(target[(i*steps):(i*steps)+steps, 1]), color='blue', marker='x', label=r"$gt_y$")
                ax.plot(t_eval, np.array(target[(i*steps):(i*steps)+steps, 2]), color='blue', marker='v', label=r"$gt_z$")
                ax.plot(t_eval, np.array(target[(i*steps):(i*steps)+steps, 3]), color='blue', marker='o', label=r"$gt_a$")

            if "test" in set.lower():
                if no_of_svariables == 2:
                    ax.plot(t_eval, np.array(pred_batch[(i*steps):(i*steps)+steps, 0]), color='r', marker='p', label=r"$pred_x$")
                    ax.plot(t_eval, np.array(pred_batch[(i*steps):(i*steps)+steps, 1]), color='r', marker='x', label=r"$pred_y$")
                elif no_of_svariables == 3:
                    ax.plot(t_eval, np.array(pred_batch[(i*steps):(i*steps)+steps, 0]), color='r', marker='p', label=r"$pred_x$")
                    ax.plot(t_eval, np.array(pred_batch[(i*steps):(i*steps)+steps, 1]), color='r', marker='x', label=r"$pred_y$")
                    ax.plot(t_eval, np.array(pred_batch[(i*steps):(i*steps)+steps, 2]), color='r', marker='v', label=r"$pred_z$")
                elif no_of_svariables == 4:
                    ax.plot(t_eval, np.array(pred_batch[(i*steps):(i*steps)+steps, 0]), color='r', marker='p', label=r"$pred_x$")
                    ax.plot(t_eval, np.array(pred_batch[(i*steps):(i*steps)+steps, 1]), color='r', marker='x', label=r"$pred_y$")
                    ax.plot(t_eval, np.array(pred_batch[(i*steps):(i*steps)+steps, 2]), color='r', marker='v', label=r"$pred_z$")
                    ax.plot(t_eval, np.array(pred_batch[(i*steps):(i*steps)+steps, 3]), color='r', marker='o', label=r"$pred_a$")

            ax.legend()
            ax.grid()
            ax.set_xlabel(r"$t$")
            if no_of_svariables == 2:
                ax.set_ylabel(r"$x(t), y(t)$")
            elif no_of_svariables == 3:
                ax.set_ylabel(r"$x(t), y(t), z(t)$")
            elif no_of_svariables == 4:
                ax.set_ylabel(r"$x(t), y(t), z(t), a(t)$")

            if no_of_svariables == 2:
                if no_of_params == 1:
                    _, x0, y0, a0 = dataset[(i*steps), :]
                    ax.set_title(fr"$x_0={x0:.2f}$, $y_0={y0:.2f}$, params=${a0:.2f}$")
                elif no_of_params == 2:
                    _, x0, y0, a0, b0 = dataset[(i*steps), :]
                    ax.set_title(fr"$x_0={x0:.2f}$, $y_0={y0:.2f}$, params=${a0:.2f}, {b0:.2f}$")
                elif no_of_params == 3:
                    _, x0, y0, a0, b0, c0 = dataset[(i*steps), :]
                    ax.set_title(fr"$x_0={x0:.2f}$, $y_0={y0:.2f}$, params=${a0:.2f}, {b0:.2f}, {c0:.2f}$")
                elif no_of_params == 4:
                    _, x0, y0, a0, b0, c0, d0 = dataset[(i*steps), :]
                    ax.set_title(fr"$x_0={x0:.2f}$, $y_0={y0:.2f}$, params=${a0:.2f}, {b0:.2f}, {c0:.2f}, {d0:.2f}$", wrap=True)
            elif no_of_svariables == 3:
                if no_of_params == 1:
                    _, x0, y0, z0, a0 = dataset[(i*steps), :]
                    ax.set_title(fr"$x_0={x0:.2f}$, $y_0={y0:.2f}$, $z_0={z0:.2f}$, params=${a0:.2f}$")
                elif no_of_params == 2:
                    _, x0, y0, z0, a0, b0 = dataset[(i*steps), :]
                    ax.set_title(fr"$x_0={x0:.2f}$, $y_0={y0:.2f}$, $z_0={z0:.2f}$, params=${a0:.2f}, {b0:.2f}$")
                elif no_of_params == 3:
                    _, x0, y0, z0, a0, b0, c0 = dataset[(i*steps), :]
                    ax.set_title(fr"$x_0={x0:.2f}$, $y_0={y0:.2f}$, $z_0={z0:.2f}$, params=${a0:.2f}, {b0:.2f}, {c0:.2f}$")
                elif no_of_params == 4:
                    _, x0, y0, z0, a0, b0, c0, d0 = dataset[(i*steps), :]
                    ax.set_title(fr"$x_0={x0:.2f}$, $y_0={y0:.2f}$, $z_0={z0:.2f}$, params=${a0:.2f}, {b0:.2f}, {c0:.2f}, {d0:.2f}$", wrap=True)
            elif no_of_svariables == 4:
                if no_of_params == 6:
                    _, x0, y0, z0, a0, b0, c0, d0, e0, f0, g0 = dataset[(i*steps), :]
                    ax.set_title(fr"$sv={x0:.2f}, {y0:.2f}, {z0:.2f}, {a0:.2f}, \theta={b0:.2f}, {c0:.2f}, {d0:.2f}, {e0:.2f}, {f0:.2f}, {g0:.2f}$", wrap=True)

        plt.tight_layout()
        plt.savefig(f'figs/{set}/{name}.png', dpi=300)
        plt.close()


    def plot_creations_3d(self, t_eval, dataset, target, steps, no_plot_figures=10, set: str="train", pred_batch=None):
        """Function to plot the target dataset and the original dataset"""
        for i in range(no_plot_figures):
            plt.figure(figsize=(10, 5))

            plt.plot(t_eval, np.array(target[(i*steps):(i*steps)+steps, 0]), color='blue', marker='x', label=r"$gt_x$")
            plt.plot(t_eval, np.array(target[(i*steps):(i*steps)+steps, 1]), color='blue', marker='o', label=r"$gt_y$")
            plt.plot(t_eval, np.array(target[(i*steps):(i*steps)+steps, 2]), color='blue', marker='v', label=r"$gt_z$")

            if "test" in set.lower():
                plt.plot(t_eval, np.array(pred_batch[(i*steps):(i*steps)+steps, 0]), color='r', marker='x', label=r"$pred_x$")
                plt.plot(t_eval, np.array(pred_batch[(i*steps):(i*steps)+steps, 1]), color='r', marker='o', label=r"$pred_y$")
                plt.plot(t_eval, np.array(pred_batch[(i*steps):(i*steps)+steps, 2]), color='r', marker='v', label=r"$pred_z$")

            plt.legend()
            plt.grid()
            plt.xlabel(r"$t$")
            plt.ylabel(r"$x(t), y(t)$")
            _, x0, y0, z0, _, _, _, _ = dataset[(i*steps), :]
            plt.title(fr"Prediction for $x_0={x0:f}$, $y_0={y0:f}$, $z_0={z0:f}$")

            plt.savefig(f'figs/{set}/{set}_{i}.png', dpi=300)
            plt.close()


    def plot_error_distributions(self, preds, gt, set, name, no_of_svariables):
        diff = abs(preds - gt)

        if no_of_svariables == 2:
            _, axs = plt.subplots(1, 2, figsize=(12, 5))
            sns.histplot(diff[:, 0], ax=axs[0], alpha=0.5, bins=30, kde=True, fill=True, color="green", label=r"$x_{\text{error}}$")
            axs[0].axvline(diff[:, 0].mean(), color='r', linestyle='--', label=rf"$\mu={diff[:, 0].mean():.2f}$")
            axs[0].legend()
            axs[0].set_title(r"Absolute Error distribution in $X_{error}$")
            axs[0].set_xlabel("Error")
            sns.histplot(diff[:, 1], ax=axs[1], kde=True, alpha=0.5, bins=30, label=r"$y_{\text{error}}$")
            axs[1].axvline(diff[:, 1].mean(), color='r', linestyle='--', label=rf"$\mu={diff[:, 1].mean():.2f}$")
            axs[1].legend()
            axs[1].set_title(r"Absolute Error distribution in $Y_{error}$")
            axs[1].set_xlabel("Error")
            axs[0].grid()
            axs[1].grid()
            plt.legend()
            plt.tight_layout()

            plt.savefig(f"figs/{set}/error_distribution_{name}.png", dpi=300)
            plt.close()
        elif no_of_svariables == 3:
            _, axs = plt.subplots(1, 3, figsize=(12, 5))
            sns.histplot(diff[:, 0], ax=axs[0], alpha=0.5, bins=30, kde=True, fill=True, color="green", label=r"$x_{\text{error}}$")
            axs[0].axvline(diff[:, 0].mean(), color='r', linestyle='--', label=rf"$\mu={diff[:, 0].mean():.2f}$")
            axs[0].legend()
            axs[0].set_title(r"Absolute Error distribution in $X_{error}$")
            axs[0].set_xlabel("Error")
            sns.histplot(diff[:, 1], ax=axs[1], kde=True, alpha=0.5, bins=30, label=r"$y_{\text{error}}$")
            axs[1].axvline(diff[:, 1].mean(), color='r', linestyle='--', label=rf"$\mu={diff[:, 1].mean():.2f}$")
            axs[1].legend()
            axs[1].set_title(r"Absolute Error distribution in $Y_{error}$")
            axs[1].set_xlabel("Error")
            sns.histplot(diff[:, 2], ax=axs[2], kde=True, alpha=0.5, bins=30, label=r"$y_{\text{error}}$")
            axs[2].axvline(diff[:, 2].mean(), color='r', linestyle='--', label=rf"$\mu={diff[:, 2].mean():.2f}$")
            axs[2].legend()
            axs[2].set_title(r"Absolute Error distribution in $Y_{error}$")
            axs[2].set_xlabel("Error")
            axs[0].grid()
            axs[1].grid()
            axs[2].grid()
            plt.legend()
            plt.tight_layout()

            plt.savefig(f"figs/{set}/error_distribution_{name}.png", dpi=300)
            plt.close()