import jax
import jax.numpy as jnp
from time import time
from functools import partial
import numpy as np
from .neural_network import batched_prediction # Import batched_prediction
from .system_definition import return_func_output, create_dynamical_system # Import necessary functions
from scipy.integrate import solve_ivp # Keep solve_ivp for generating target data

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

@jax.jit
def mse_loss(predictions, targets):
    """ Data/Mean squared error loss of the neural network model """
    # Ensure predictions and targets have the same shape for element-wise operations
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape for MSE loss.")
    diff = predictions - targets
    return jnp.mean(diff*diff) # Use mean to handle arbitrary dimensions

@jax.jit
def rmse_loss(predictions, targets):
    """ Root mean squared error loss of the neural network model """
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape for RMSE loss.")
    diff = predictions - targets
    return jnp.sqrt(jnp.mean(diff*diff))

@jax.jit
def mae_loss(predictions, targets):
    """ Mean absolute error loss of the neural network model """
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape for MAE loss.")
    diff = jnp.abs(predictions - targets)
    return jnp.mean(diff)

# @jax.jit
def compute_jacobian(params, activations):
    """ Computing Jacobians of the MLP feature vectors w.r.t. time t """
    # Assuming the first feature in activations is time (index 0)
    # The output shape of batched_prediction is (batch_size, num_targets)
    # We want the derivative of each target with respect to time for each batch element.
    # The Jacobian will have shape (batch_size, num_targets, num_features)
    # Since each model predicts a single state variable, num_targets is 1 for each model.
    # The output shape of batched_prediction(params[i], activations, ...) is (batch_size, 1).
    # The jacobian shape is (batch_size, 1, num_features).

    time_derivatives = []
    for param_set in params:
        # Compute the jacobian for this state variable's model
        jac = jax.jacfwd(batched_prediction, argnums=1)(param_set, activations, jax.nn.silu)
        # Extract the derivative with respect to the time feature (index 0 in the last dimension)
        time_derivative = jac[:, :, :, 0].sum(axis=-1) # Shape (batch_size,)
        time_derivatives.append(time_derivative)

    # Stack the time derivatives to get an array of shape (batch_size, num_state_vars)
    return jnp.stack(time_derivatives, axis=1).squeeze()

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
def combined_loss(model, activations, initial_condition_indices, pst, lpst, t3, t4, ft_funcs, fot_funcs, system_args, alpha=1.0, beta=1.0):
    """ Combined loss function that combines the MSE loss and the time-coupled loss """
    num_state_vars = len(model) # Number of state variables is the number of separate models

    # Get predictions for all state variables
    predictions = jnp.stack([jnp.ravel(batched_prediction(model[i], activations, jax.nn.silu)) for i in range(num_state_vars)], axis=1) # Shape (batch_size, num_state_vars)

    # Compute time derivatives of predictions
    diff_predictions = compute_jacobian(model, activations) # Shape (batch_size, num_state_vars)

    # Calculate the left-hand side of the time-coupled loss
    lterm = lpst + t3[:, None] * predictions + t4[:, None] * diff_predictions # Use broadcasting with None

    # Construct the state for the original ODE function with current predictions.
    # The original activations contain time, initial state variables, and parameters.
    # We need to replace the initial state variable columns with the current predictions.
    # The time column remains the same, and the parameter columns remain the same.
    # activations shape: (batch_size, 1 + num_state_vars + num_params)
    # predictions shape: (batch_size, num_state_vars)
    current_state_for_ode = activations.at[:, 1:1+num_state_vars].set(predictions) # Update state variable columns

    # Calculate the right-hand side of the time-coupled loss using the original ODEs
    # ft_funcs contains the lambdified original ODEs
    # Each ODE function takes (state_var_1, ..., state_var_n, param_1, ..., param_m) as arguments.
    # The 'args' for return_func_output should match the order of these symbols.
    # The input 'state' to return_func_output should contain the values for these symbols.
    # current_state_for_ode[:, 1:] has shape (batch_size, num_state_vars + num_params)
    # and contains [predicted_state_var_1, ..., predicted_state_var_n, param_1_val, ..., param_m_val].
    # The order matches system_args which is [state_var_symbols..., param_symbols...].
    rterm = jnp.stack([return_func_output(eqn_num=i, state=current_state_for_ode[:, 1:], func=ft_funcs, args=system_args) for i in range(num_state_vars)], axis=1) # Shape (batch_size, num_state_vars)

    # Handle initial conditions separately
    initial_activations = activations[initial_condition_indices]
    reminder_init_lterm = predictions[initial_condition_indices, :] # Shape (num_initial_conditions, num_state_vars)

    # For initial conditions, the state variables are taken directly from the initial_activations.
    # initial_activations has shape (num_initial_conditions, 1 + num_state_vars + num_params)
    # The state variables and parameters at initial conditions are in columns 1 onwards.
    initial_state_for_ode = initial_activations[:, 1:] # Shape (num_initial_conditions, num_state_vars + num_params)

    # reminder_init_rterm is the value of the first time derivative at t=0 (using fot_funcs, which are the same as ft_funcs)
    # fot_funcs contains the lambdified first time derivatives (same as ft_funcs)
    # return_func_output expects state with shape (batch_size, num_state_vars + num_params) and args as system_args.
    reminder_init_rterm = jnp.stack([return_func_output(eqn_num=i, state=initial_state_for_ode, func=fot_funcs, args=system_args) for i in range(num_state_vars)], axis=1) # Shape (num_initial_conditions, num_state_vars)

    # Compute the overall loss
    # The MSE loss is computed across all state variables for both the main term and the initial condition term.
    main_loss = mse_loss(lterm, rterm)
    initial_condition_loss = mse_loss(reminder_init_lterm, reminder_init_rterm)

    return alpha * main_loss + beta * initial_condition_loss


@partial(jax.jit, static_argnames=['ft_funcs', 'fot_funcs', 'system_args'])
def adam_update_gd(model, adam_state, activations, initial_condition_indices, pst, lpst, t3, t4, lr, ft_funcs, fot_funcs, system_args, alpha=1.0, beta=1.0):
    """ Update the parameters of the model using the Adam optimizer on the combined loss function """
    t = adam_state["t"] + 1

    total_loss_grads = jax.grad(combined_loss)(model, activations, initial_condition_indices, pst, lpst, t3, t4, ft_funcs, fot_funcs, system_args, alpha, beta)
    updated_params, new_adam_state = update_adam_internal_state(adam_state, total_loss_grads, model, lr, t)

    return updated_params, new_adam_state

def get_intr_time_coupled_sums(activations, steps, t, t2, t3, ft_funcs, st_funcs, tt_funcs, args):
    """ Computes intermediate terms for the time-coupled loss """
    # Assuming activations shape is [batch_size * steps, 1 + num_state_vars + num_params]
    # The first column is time, followed by initial state variables and then parameters.
    num_state_vars = len(ft_funcs) # Number of state variables

    # Extract initial conditions and parameters for each trajectory in the batch
    # These are in columns 1 onwards of the activations array at the first time step of each trajectory.
    initial_conditions_batch = activations[::steps, 1:] # shape [batch_size, num_state_vars + num_params]

    # Compute the time derivatives at the initial conditions for each state variable.
    # The derivative functions (ft_funcs, st_funcs, tt_funcs) are a list, one function per state variable.
    # Each function takes (state_var_1, ..., state_var_n, param_1, ..., param_m) as arguments.
    # return_func_output applies the function to the initial_conditions_batch.
    # The 'args' parameter for return_func_output should match the order of symbols expected by the lambdified functions.
    # Since the functions take state variables and parameters, 'args' should be system_args.
    intr_2_batch = jnp.stack([return_func_output(eqn_num=i, state=initial_conditions_batch, func=ft_funcs, args=args) for i in range(num_state_vars)], axis=1) # Shape (batch_size, num_state_vars)
    intr_3_batch = jnp.stack([return_func_output(eqn_num=i, state=initial_conditions_batch, func=st_funcs, args=args) for i in range(num_state_vars)], axis=1) # Shape (batch_size, num_state_vars)
    intr_4_batch = jnp.stack([return_func_output(eqn_num=i, state=initial_conditions_batch, func=tt_funcs, args=args) for i in range(num_state_vars)], axis=1) # Shape (batch_size, num_state_vars)

    # Repeat these initial condition derivative values for all steps in each trajectory
    intr_2 = jnp.repeat(intr_2_batch, steps, axis=0) # Shape (batch_size * steps, num_state_vars)
    intr_3 = jnp.repeat(intr_3_batch, steps, axis=0) # Shape (batch_size * steps, num_state_vars)
    intr_4 = jnp.repeat(intr_4_batch, steps, axis=0) # Shape (batch_size * steps, num_state_vars)

    # Extract initial state variable values for each trajectory at each time step
    # These are in columns 1 to num_state_vars of the activations array.
    # Since sample_generation repeats the initial conditions for each time step,
    # activations[:, 1:1+num_state_vars] already contains the initial state variables
    # repeated for all time steps in each trajectory.
    term_1 = activations[:, 1:1+num_state_vars] # Shape (batch_size * steps, num_state_vars)

    # Apply time terms using broadcasting
    term_2 = t[:, None] * intr_2  # Shape (batch_size * steps, num_state_vars)
    term_3 = t2[:, None] * intr_3 # Shape (batch_size * steps, num_state_vars)
    term_4 = t3[:, None] * intr_4 # Shape (batch_size * steps, num_state_vars)

    ps_tc = term_1 + term_2 + term_3 + term_4
    lps_tc = intr_2 + t[:, None] * intr_3 + t2[:, None] * intr_4

    return ps_tc, lps_tc

def get_time_terms(batch_size, t_eval):
    """ Computes time-related terms for the loss function """
    train_t = jnp.tile(t_eval, reps=batch_size)
    train_t2 = (train_t**2)/2
    train_t3 = (train_t**3)/6
    train_t4 = (train_t**4)/24

    return train_t, train_t2, train_t3, train_t4

def sample_generation(s_size, steps, t_eval, initial_conditions_range, parameter_ranges, key):
    """
    Generates sample data for training and validation.

    Args:
        s_size (int): Number of unique initial conditions.
        steps (int): Number of time steps.
        t_eval (jnp.ndarray): Array of time points.
        initial_conditions_range (list): List of tuples, where each tuple is (min_val, max_val) for a state variable.
        parameter_ranges (list): List of tuples, where each tuple is (min_val, max_val) for a parameter.
        key (int): Random seed.
        num_state_vars (int): Number of state variables.
        num_params (int): Number of parameters.
    Returns:
        jnp.ndarray: Generated dataset, shape (s_size * steps, 1 + num_state_vars + num_params). Columns are [time, state_var_1_init, ..., state_var_n_init, param_1, ..., param_m].
    """
    ran_key = jax.random.PRNGKey(key)
    init_conds = []
    for min_val, max_val in initial_conditions_range:
        key, subkey = jax.random.split(ran_key)
        init_conds.append(jax.random.uniform(subkey, minval=min_val, maxval=max_val, shape=(s_size, )))

    params = []
    for min_val, max_val in parameter_ranges:
        key, subkey = jax.random.split(ran_key)
        params.append(jax.random.uniform(subkey, minval=min_val, maxval=max_val, shape=(s_size, )))

    # Stack initial conditions and parameters
    initial_state_and_params = jnp.stack(init_conds + params, axis=1) # Shape (s_size, num_state_vars + num_params)

    # Repeat for each time step
    repeated_initial_state_and_params = initial_state_and_params.repeat(steps, axis=0) # Shape (s_size * steps, num_state_vars + num_params)

    # Tile time evaluation points
    t_eval_set = jnp.tile(t_eval, reps=s_size).reshape(-1, 1) # Shape (s_size * steps, 1)

    # Concatenate time and the repeated initial state and parameters
    dataset = jnp.concat([t_eval_set, repeated_initial_state_and_params], axis=1) # Shape (s_size * steps, 1 + num_state_vars + num_params)

    return dataset


def get_batch_data(dataset, steps, t_eval, training_batch_size):
    """ Get a randomly shuffled batch data for training the model """
    # Select random initial condition indices from the dataset where time is 0
    initial_condition_indices = jnp.where(dataset[:, 0] == 0)[0] # Indices in the original dataset where t=0

    # Randomly select a batch of these initial condition indices
    # Use numpy for sampling if jax.random.choice with replace=False is not suitable
    random_initial_batch_indices = np.random.choice(initial_condition_indices, size=training_batch_size, replace=False)

    # For each selected initial condition index, get the indices of all its time steps
    batch_indices = []
    # Infer original number of steps by finding the number of unique time points for a single trajectory
    unique_times_per_trajectory = jnp.unique(dataset[:, 0]).shape[0]
    original_steps = unique_times_per_trajectory

    for idx in random_initial_batch_indices:
        # Assuming the data is ordered by initial condition, then time
        start_idx_in_dataset = idx # The index in the original dataset corresponds to the t=0 point
        end_idx_in_dataset = start_idx_in_dataset + original_steps
        batch_indices.extend(range(start_idx_in_dataset, end_idx_in_dataset))

    # Select the batch data from the original dataset
    random_batch_of_features = dataset[jnp.array(batch_indices), :]

    # Ensure the time column is correctly set based on t_eval for the batch
    # The get_time_terms function already creates the tiled t_eval
    t_eval_reshaped = jnp.tile(t_eval, reps=training_batch_size)
    random_batch_of_features = random_batch_of_features.at[:, 0].set(t_eval_reshaped)


    return random_batch_of_features

def define_ode_target(initial_conditions_batch, t_eval, duration, ode_system_func, parameter_symbols):
    """
    Generates target data by numerically solving the ODE for a batch of initial conditions.

    Args:
        initial_conditions_batch (jnp.ndarray): Batch of initial conditions (state variables and parameters),
                                                 shape (batch_size, num_state_vars + num_params).
        t_eval (jnp.ndarray): Array of time points, shape (steps,).
        duration (float): The duration of the simulation.
        ode_system_func (function): The function defining the ODE system for solve_ivp.
        parameter_symbols (list): List of SymPy symbols for parameters.

    Returns:
        jnp.ndarray: The numerical solution of the ODE for the given initial conditions and time points,
                     shape (batch_size * steps, num_state_vars).
    """
    solution_set = []
    num_state_vars = initial_conditions_batch.shape[1] - len(parameter_symbols)

    for initial_cond_with_params in initial_conditions_batch:
        # Separate initial state variables and parameters
        initial_state_vars = initial_cond_with_params[:num_state_vars]
        params_vals = initial_cond_with_params[num_state_vars:]

        sol = solve_ivp(ode_system_func, t_span=[0, duration], t_eval=t_eval, y0=initial_state_vars, args=tuple(params_vals), method='RK45')
        # sol.y has shape (num_state_vars, steps). Transpose to get (steps, num_state_vars)
        solution_set.extend(sol.y.T)

    return jnp.array(solution_set)

def rel_l2_error(pred, true):
    """ Compute the relative L2 error between the predicted and true values """
    # Compute the L2 norm over the flattened arrays.
    pred_flat = pred.ravel()
    true_flat = true.ravel()

    # Avoid division by zero if true is all zeros
    true_norm = jnp.sqrt(jnp.sum(true_flat**2))
    if true_norm < 1e-8: # Add a small tolerance
        return jnp.array(jnp.nan) # Or handle as an error case

    return jnp.sqrt(jnp.sum((pred_flat - true_flat)**2)) / true_norm