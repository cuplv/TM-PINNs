import sympy as sp
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

def create_dynamical_system(symbols, equations, state_vars_symbols):
    num_state_vars = len(state_vars_symbols)
    state_vars = symbols[:num_state_vars]
    params = symbols[num_state_vars:]
    args = state_vars + params
    assert len(state_vars) + len(params) == len(args)

    F_dot = sp.Matrix(equations)

    def compute_derivative(matrix, state_vars, equations):
        """
        Computes the time derivative of a symbolic matrix expression.
        """
        return sp.Matrix([sum(sp.diff(matrix[i], var) * eq for var, eq in zip(state_vars, equations)) for i in range(len(matrix))])

    F_ddot = compute_derivative(F_dot, state_vars, equations)
    F_dddot = compute_derivative(F_ddot, state_vars, equations)
    F_ddddot = compute_derivative(F_dddot, state_vars, equations)

    funcs = [sp.lambdify(args, e, modules='jax') for e in F_dot]
    deriv_funcs = [sp.lambdify(args, e, modules='jax') for e in F_ddot]
    dderiv_funcs = [sp.lambdify(args, e, modules='jax') for e in F_dddot]
    ddderiv_funcs = [sp.lambdify(args, e, modules='jax') for e in F_ddddot]

    print("First derivative: ", F_dot)
    print("Second derivative: ", F_ddot)
    print("Third derivative: ", F_dddot)
    print("Fourth derivative: ", F_ddddot)

    return funcs, deriv_funcs, dderiv_funcs, ddderiv_funcs, args

def define_ode_target(initial_conditions_batch, t_eval, duration, ode_system_func, parameter_symbols):
    solution_set = []
    num_state_vars = initial_conditions_batch.shape[1] - len(parameter_symbols)

    for initial_cond_with_params in initial_conditions_batch:
        initial_state_vars = initial_cond_with_params[:num_state_vars]
        params_vals = initial_cond_with_params[num_state_vars:]

        sol = solve_ivp(ode_system_func, t_span=[0, duration], t_eval=t_eval, y0=initial_state_vars, args=tuple(params_vals), method='RK45')
        solution_set.extend(sol.y.T)

    return jnp.array(solution_set)

def get_random_key(key):
    """ Get random function key from initial random key """
    _, func_key = jax.random.split(jax.random.PRNGKey(key))
    return func_key

def sample_generation(s_size, steps, t_eval, initial_conditions_range, parameter_ranges, key, keyadd):
    init_conds = []
    for ind, z in enumerate(initial_conditions_range):
        min_val, max_val = z
        init_conds.append(jax.random.uniform(get_random_key(key+keyadd[ind]), minval=min_val, maxval=max_val, shape=(s_size, )))

    params = []
    for ind2, z in enumerate(parameter_ranges):
        min_val, max_val = z
        params.append(jax.random.uniform(get_random_key(key+keyadd[ind+ind2+1]), minval=min_val, maxval=max_val, shape=(s_size, )))

    initial_state_and_params = jnp.stack(init_conds + params, axis=1) # Shape (s_size, num_state_vars + num_params)
    repeated_initial_state_and_params = initial_state_and_params.repeat(steps, axis=0) # Shape (s_size * steps, num_state_vars + num_params)
    t_eval_set = jnp.tile(t_eval, reps=s_size).reshape(-1, 1) # Shape (s_size * steps, 1)
    dataset = jnp.concat([t_eval_set, repeated_initial_state_and_params], axis=1) # Shape (s_size * steps, 1 + num_state_vars + num_params)

    return dataset

def define_ode_system(symbols, equations, state_vars_symbols):
    num_state_vars = len(state_vars_symbols)
    state_vars = symbols[:num_state_vars]
    params = symbols[num_state_vars:]
    args = state_vars + params
    assert len(state_vars) + len(params) == len(args)
    
    diffEq_func = [sp.lambdify(args, e, modules="jax") for e in equations]
    def Eqn(t, y, *params_vals):
        all_args = list(y) + list(params_vals)
        dydt = [diffEq_func[i](*all_args) for i in range(len(diffEq_func))]
        return dydt

    return Eqn

def get_time_terms(batch_size, t_eval):
    """ Computes time-related terms for the loss function """
    train_t = jnp.tile(t_eval, reps=batch_size)
    train_t2 = (train_t**2)/2
    train_t3 = (train_t**3)/6
    train_t4 = (train_t**4)/24

    return train_t, train_t2, train_t3, train_t4

def get_batch_data(dataset, steps, t_eval, training_batch_size):
    """ Get a randomly shuffled batch data for training the model """
    train_init_indices = jnp.where(jnp.isclose(dataset[:, 0], 0.0, atol=1e-8))[0]

    if train_init_indices.size == 0:
        # Print debug info and raise error before proceeding
        print("DEBUG: Dataset times:", jnp.unique(dataset[:, 0]))
        raise ValueError("No initial condition indices found for time = 0 in dataset")
    
    train_init_indices = np.random.choice(train_init_indices, size=training_batch_size, replace=False) # Randomly select init training indices half the size of the batch
    random_batch_of_features = dataset[train_init_indices, :]
    random_batch_of_features = random_batch_of_features.repeat(steps, axis=0)
    t_eval = jnp.tile(t_eval, reps=training_batch_size)
    random_batch_of_features = random_batch_of_features.at[:, 0].set(t_eval)
    
    return random_batch_of_features

def return_func_output(eqn_num, state, func, args):
    vmap_args = tuple(state[:, i] for i in range(state.shape[1]))
    return jax.vmap(lambda *vmap_args: func[eqn_num](*vmap_args))(*vmap_args)