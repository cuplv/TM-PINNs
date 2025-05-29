import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from .neural_network import feedforward_prediction
from .system_definition import return_func_output

beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

def initialize_adam_state(params):
    """ Initialize Adam state for the parameters """
    adam_state = {
        "m": jax.tree_util.tree_map(jnp.zeros_like, params),
        "v": jax.tree_util.tree_map(jnp.zeros_like, params),
        "t": 0  # Time step
    }
    return adam_state

def batched_prediction(params, activations, fn):
    return jax.vmap(feedforward_prediction, in_axes=(None, 0, None))(params, activations, fn)

@partial(jax.jit, static_argnames=["steps", "ft_funcs", "st_funcs", "tt_funcs", "args"])
def get_intr_time_coupled_sums(activations, steps, t, t2, t3, ft_funcs, st_funcs, tt_funcs, args):
    """ Computes intermediate terms for the time-coupled loss """
    num_state_vars = len(ft_funcs) # Number of state variables
    initial_conditions_batch = activations[::steps, 1:] # shape [batch_size, num_state_vars + num_params]

    intr_2_batch = jnp.stack([return_func_output(eqn_num=i, state=initial_conditions_batch, func=ft_funcs, args=args) for i in range(num_state_vars)], axis=1)
    intr_3_batch = jnp.stack([return_func_output(eqn_num=i, state=initial_conditions_batch, func=st_funcs, args=args) for i in range(num_state_vars)], axis=1) 
    intr_4_batch = jnp.stack([return_func_output(eqn_num=i, state=initial_conditions_batch, func=tt_funcs, args=args) for i in range(num_state_vars)], axis=1) 

    intr_2 = jnp.repeat(intr_2_batch, steps, axis=0) # Shape (batch_size * steps, num_state_vars)
    intr_3 = jnp.repeat(intr_3_batch, steps, axis=0) # Shape (batch_size * steps, num_state_vars)
    intr_4 = jnp.repeat(intr_4_batch, steps, axis=0) # Shape (batch_size * steps, num_state_vars)

    term_1 = activations[:, 1:1+num_state_vars] # Shape (batch_size * steps, num_state_vars)

    term_2 = t[:, None] * intr_2  # Shape (batch_size * steps, num_state_vars)
    term_3 = t2[:, None] * intr_3 # Shape (batch_size * steps, num_state_vars)
    term_4 = t3[:, None] * intr_4 # Shape (batch_size * steps, num_state_vars)

    ps_tc = term_1 + term_2 + term_3 + term_4
    lps_tc = intr_2 + t[:, None] * intr_3 + t2[:, None] * intr_4

    return ps_tc, lps_tc

@jax.jit
def mse_loss(predictions, targets):
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape for MSE loss.")
    diff = predictions - targets
    return jnp.mean(diff*diff) # Use mean to handle arbitrary dimensions

@jax.jit
def rmse_loss(predictions, targets):
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape for RMSE loss.")
    diff = predictions - targets
    return jnp.sqrt(jnp.mean(diff*diff))

@jax.jit
def mae_loss(predictions, targets):
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape for MAE loss.")
    diff = jnp.abs(predictions - targets)
    return jnp.mean(diff)

@jax.jit
def compute_jacobian(params, activations):
    time_derivatives = []
    for param_set in params:
        # Compute the jacobian for this state variable's model
        jac = jax.jacfwd(batched_prediction, argnums=1)(param_set, activations, jax.nn.silu)
        # Extract the derivative with respect to the time feature (index 0 in the last dimension)
        time_derivative = jac[:, :, :, 0].sum(axis=-1) # Shape (batch_size,)
        time_derivatives.append(time_derivative)

    return jnp.stack(time_derivatives, axis=1).squeeze()

@partial(jax.jit, static_argnames=['ft_funcs', 'fot_funcs', 'system_args'])
def combined_loss(model, activations, initial_condition_indices, pst, lpst, t3, t4, ft_funcs, fot_funcs, system_args, alpha=1.0, beta=1.0):
    """ Combined loss function that combines the MSE loss and the time-coupled loss """
    num_state_vars = len(model) # Number of state variables is the number of separate models
    predictions = jnp.stack([jnp.ravel(batched_prediction(model[i], activations, jax.nn.silu)) for i in range(num_state_vars)], axis=1) # Shape (batch_size, num_state_vars)
    diff_predictions = compute_jacobian(model, activations) # Shape (batch_size, num_state_vars)
    
    lterm = lpst + t3[:, None] * predictions + t4[:, None] * diff_predictions # Use broadcasting with None
    current_state_for_ode = activations.at[:, 1:1+num_state_vars].set(predictions) # Update state variable columns

    rterm = jnp.stack([return_func_output(eqn_num=i, state=current_state_for_ode[:, 1:], func=ft_funcs, args=system_args) for i in range(num_state_vars)], axis=1) # Shape (batch_size, num_state_vars)

    initial_activations = activations[initial_condition_indices]
    reminder_init_lterm = predictions[initial_condition_indices, :] # Shape (num_initial_conditions, num_state_vars)
    initial_state_for_ode = initial_activations[:, 1:] # Shape (num_initial_conditions, num_state_vars + num_params)
    reminder_init_rterm = jnp.stack([return_func_output(eqn_num=i, state=initial_state_for_ode, func=fot_funcs, args=system_args) for i in range(num_state_vars)], axis=1) # Shape (num_initial_conditions, num_state_vars)

    main_loss = mse_loss(lterm, rterm)
    initial_condition_loss = mse_loss(reminder_init_lterm, reminder_init_rterm)

    return alpha * main_loss + beta * initial_condition_loss

@jax.jit
def update_adam_internal_state(adam_state, total_loss_grads, params, lr, t):
    new_m = jax.tree_util.tree_map(lambda m, g: beta1 * m + (1 - beta1) * g, adam_state["m"], total_loss_grads)
    new_v = jax.tree_util.tree_map(lambda v, g: beta2 * v + (1 - beta2) * (g ** 2), adam_state["v"], total_loss_grads)

    m_hat = jax.tree_util.tree_map(lambda m: m / (1 - beta1 ** t), new_m)
    v_hat = jax.tree_util.tree_map(lambda v: v / (1 - beta2 ** t), new_v)

    updated_params = jax.tree_util.tree_map(lambda p, m, v: p - lr * m / (jnp.sqrt(v) + epsilon), params, m_hat, v_hat)

    new_adam_state = {"m": new_m, "v": new_v, "t": t}
    return updated_params, new_adam_state

@partial(jax.jit, static_argnames=['ft_funcs', 'fot_funcs', 'system_args'])
def adam_update_gd(model, adam_state, activations, initial_condition_indices, pst, lpst, t3, t4, lr, ft_funcs, fot_funcs, system_args, alpha=1.0, beta=1.0):
    """ Update the parameters of the model using the Adam optimizer on the combined loss function """
    t = adam_state["t"] + 1

    total_loss_grads = jax.grad(combined_loss)(model, activations, initial_condition_indices, pst, lpst, t3, t4, ft_funcs, fot_funcs, system_args, alpha, beta)
    updated_params, new_adam_state = update_adam_internal_state(adam_state, total_loss_grads, model, lr, t)

    return updated_params, new_adam_state
