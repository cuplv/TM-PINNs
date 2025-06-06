import os
import jax
import jax.numpy as jnp
from jax.nn import silu
import numpy as np
from copy import deepcopy
from time import time
from utils import Helpers
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import click

os.environ["ENABLE_PJRT_COMPATIBILITY"] = "1"
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

def get_random_key(key):
    """ Get random function key from initial random key """
    _, func_key = jax.random.split(jax.random.PRNGKey(key))
    return func_key

def initialize_adam_state(params, num_var):
    """ Initialize Adam state for the parameters """
    adam_state = {
        "m": [[(jnp.zeros_like(w), jnp.zeros_like(b)) for w, b in params[i]] for i in range(num_var)],
        "v": [[(jnp.zeros_like(w), jnp.zeros_like(b)) for w, b in params[i]] for i in range(num_var)],
        "t": 0  # Time step
    }
    return adam_state

def feedforward_prediction(params, activations, fn):
    """ Feedforward neural network model that applies the activation function on the computation of inputs to the next layer. """
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = fn(outputs)
    
    w_final, b_final = params[-1]
    final_op = jnp.dot(w_final, activations) + b_final
    return final_op

@jax.jit
def mse_loss(predictions, targets):
    """ Data/Mean squared error loss of the neural network model """
    diff = predictions - targets
    return jnp.sum(diff*diff)/predictions.shape[0]

@jax.jit
def rmse_loss(predictions, targets):
    """ Root mean squared error loss of the neural network model """
    diff = predictions - targets
    return jnp.sqrt(jnp.sum(diff*diff)/predictions.shape[0])

@jax.jit
def mae_loss(predictions, targets):
    """ Mean absolute error loss of the neural network model """
    diff = jnp.abs(predictions - targets)
    return jnp.mean(diff)

@jax.jit
def compute_jacobian(params, activations):
    """ Computing Jacobians of the MLP feature vectors w.r.t. time t """
    jacbs = jax.jacfwd(batched_prediction, argnums=1)(params, activations, silu) # batch_size x num_targets x batch_size x num_features
    return jacbs[:, :, :, 0].sum(axis=-1) # batch_size x time_feature

@jax.jit
def update_adam_internal_state(adam_state, total_loss_grads, params, lr, t):
    new_m = [jax.tree_util.tree_map(lambda m, g: beta1 * m + (1 - beta1) * g, adam_state["m"][i], total_loss_grads[i]) for i in range(len(adam_state["m"]))]
    new_v = [jax.tree_util.tree_map(lambda v, g: beta2 * v + (1 - beta2) * (g ** 2), adam_state["v"][i], total_loss_grads[i]) for i in range(len(adam_state["v"]))]

    m_hat = jax.tree_util.tree_map(lambda m: m / (1 - beta1 ** t), new_m)
    v_hat = jax.tree_util.tree_map(lambda v: v / (1 - beta2 ** t), new_v)

    updated_params = jax.tree_util.tree_map(lambda p, m, v: p - lr * m / (jnp.sqrt(v) + epsilon), params, m_hat, v_hat)

    new_adam_state = {"m": new_m, "v": new_v, "t": t}
    return updated_params, new_adam_state

@jax.jit
def combined_loss(model, activations, pst_x, pst_y, pst_z, pst_a, lpst_x, lpst_y, lpst_z, lpst_a, t3, t4, alpha=1.0, beta=1.0):
    """ Combined loss function that combines the MSE loss and the time-coupled loss """
    prediction_x = jnp.ravel(batched_prediction(model[0], activations, silu))
    prediction_y = jnp.ravel(batched_prediction(model[1], activations, silu))
    prediction_z = jnp.ravel(batched_prediction(model[2], activations, silu))
    prediction_a = jnp.ravel(batched_prediction(model[3], activations, silu))
    diff_prediction_x = jnp.ravel(compute_jacobian(model[0], activations))
    diff_prediction_y = jnp.ravel(compute_jacobian(model[1], activations))
    diff_prediction_z = jnp.ravel(compute_jacobian(model[2], activations))
    diff_prediction_a = jnp.ravel(compute_jacobian(model[3], activations))

    lterm_x = lpst_x + t3 * prediction_x + t4 * diff_prediction_x
    lterm_y = lpst_y + t3 * prediction_y + t4 * diff_prediction_y
    lterm_z = lpst_z + t3 * prediction_z + t4 * diff_prediction_z
    lterm_a = lpst_a + t3 * prediction_a + t4 * diff_prediction_a

    finalrterm_x = t4 * prediction_x
    finalrterm_y = t4 * prediction_y
    finalrterm_z = t4 * prediction_z
    finalrterm_a = t4 * prediction_a

    rterm_x = pst_x + finalrterm_x
    rterm_y = pst_y + finalrterm_y
    rterm_z = pst_z + finalrterm_z
    rterm_a = pst_a + finalrterm_a

    activations = activations.at[:, 1].set(prediction_x)
    activations = activations.at[:, 2].set(prediction_y)
    activations = activations.at[:, 3].set(prediction_z)
    activations = activations.at[:, 4].set(prediction_a)
    
    rterm_x = return_func_output(eqn_num=0, state=activations[:, 1:], func=ft_funcs)
    rterm_y = return_func_output(eqn_num=1, state=activations[:, 1:], func=ft_funcs)
    rterm_z = return_func_output(eqn_num=2, state=activations[:, 1:], func=ft_funcs)
    rterm_a = return_func_output(eqn_num=3, state=activations[:, 1:], func=ft_funcs)

    return alpha*mse_loss(lterm_x, rterm_x) + beta*mse_loss(lterm_y, rterm_y) + beta*mse_loss(lterm_z, rterm_z) + beta*mse_loss(lterm_a, rterm_a)

@jax.jit
def adam_update_gd(model, adam_state, activations, pst_x, pst_y, pst_z, pst_a, lpst_x, lpst_y, lpst_z, lpst_a, t3, t4, lr, alpha=1.0, beta=1.0):
    """ Update the parameters of the model using the Adam optimizer on the combined loss function """
    t = adam_state["t"] + 1

    total_loss_grads = jax.grad(combined_loss)(model, activations, pst_x, pst_y, pst_z, pst_a, lpst_x, lpst_y, lpst_z, lpst_a, t3, t4, alpha, beta)
    updated_params, new_adam_state = update_adam_internal_state(adam_state, total_loss_grads, model, lr, t)

    return updated_params, new_adam_state

def get_ode_functions(num_derivatives=3):
    """ Create a dataset for training the model """
    # 8D coupled damped oscillator system
    x, y, a, b, c, d, e, f, mu, h = sp.symbols('x y a b c d e f mu h')
    params = [mu, h]
    state_vars = [x, y, a, b, c, d, e, f]
    eqns = [
        y,
        mu*(1-x**2)*y - x + h*(a - 2*x + e),
        b,
        mu*(1-a**2)*b - a + h*(c - 2*a + x),
        d,
        mu*(1-c**2)*d - c + h*(e - 2*c + a),
        f,
        mu*(1-e**2)*f - e + h*(x - 2*e + c),
    ]
    F_dot = sp.Matrix(eqns)
    F_ddot = sp.Matrix([
        sum([sp.diff(F_dot[v], state_vars[i]) * eqns[i] for i in range(8)]) for v in range(8)
    ])
    F_dddot = sp.Matrix([
        sum([sp.diff(F_ddot[v], state_vars[i]) * eqns[i] for i in range(8)]) for v in range(8)
    ])

    class differentialComputation:
        def __init__(self, params, state_vars, F_dot, F_ddot, F_dddot, num_derivatives):
            self.params = params
            self.state_vars = state_vars
            self.F_dot = F_dot
            self.F_ddot = F_ddot
            self.F_dddot = F_dddot
            self.num_derivatives = num_derivatives
            self.n, self.m = len(state_vars), len(params)
            self.args = self.state_vars + self.params

        def create_first_time_derivative_functions(self):
            return [sp.lambdify(self.args, e, modules='jax') for e in self.F_dot]
        def create_second_time_derivative_functions(self):
            return [sp.lambdify(self.args, e, modules='jax') for e in self.F_ddot]
        def create_third_time_derivative_functions(self):
            return [sp.lambdify(self.args, e, modules='jax') for e in self.F_dddot]

    odeClass = differentialComputation(params, state_vars, F_dot, F_ddot, F_dddot, num_derivatives)
    funcs = odeClass.create_first_time_derivative_functions()
    deriv_funcs = odeClass.create_second_time_derivative_functions()
    dderiv_funcs = odeClass.create_third_time_derivative_functions()
    print(f"Functions created {odeClass.F_dot} and {odeClass.F_ddot} and {odeClass.F_dddot}")
    return funcs, deriv_funcs, dderiv_funcs

def define_ode(init_conds, t_eval, duration):
    """ Create a dataset for training the model """
    # 8D coupled damped oscillator system
    x, y, a, b, c, d, e, f, mu, h = sp.symbols('x y a b c d e f mu h')
    eqns = [
        y,
        mu*(1-x**2)*y - x + h*(a - 2*x + e),
        b,
        mu*(1-a**2)*b - a + h*(c - 2*a + x),
        d,
        mu*(1-c**2)*d - c + h*(e - 2*c + a),
        f,
        mu*(1-e**2)*f - e + h*(x - 2*e + c),
    ]
    diffEq_func = [sp.lambdify((x, y, a, b, c, d, e, f, mu, h), expr) for expr in eqns]

    def Eqn(t, y, mu, h):
        x_, y_, a_, b_, c_, d_, e_, f_ = y
        return [f(x_, y_, a_, b_, c_, d_, e_, f_, mu, h) for f in diffEq_func]

    solution_set = []
    for _, x0, y0, a0, b0, c0, d0, e0, f0, mu0, h0 in init_conds:
        sol = solve_ivp(Eqn, t_span=[0, duration], t_eval=t_eval, y0=[x0, y0, a0, b0, c0, d0, e0, f0], args=(mu0, h0), method='RK45')
        solution_set.extend(sol.y.T)
    return jnp.array(solution_set)


def sample_generation(s_size, steps, t_eval, key):
    """ Return the training dataset for the specified sample sizes """
    # State variables (initial conditions range: -0.5 to 0.5)
    x = jax.random.uniform(get_random_key(key), minval=-0.5, maxval=0.5, shape=(s_size,))
    y = jax.random.uniform(get_random_key(key+1), minval=-0.5, maxval=0.5, shape=(s_size,))
    a = jax.random.uniform(get_random_key(key+2), minval=-0.5, maxval=0.5, shape=(s_size,))
    b = jax.random.uniform(get_random_key(key+3), minval=-0.5, maxval=0.5, shape=(s_size,))
    c = jax.random.uniform(get_random_key(key+4), minval=-0.5, maxval=0.5, shape=(s_size,))
    d = jax.random.uniform(get_random_key(key+5), minval=-0.5, maxval=0.5, shape=(s_size,))
    e = jax.random.uniform(get_random_key(key+6), minval=-0.5, maxval=0.5, shape=(s_size,))
    f = jax.random.uniform(get_random_key(key+7), minval=-0.5, maxval=0.5, shape=(s_size,))
    # Parameters (mu, h) (range: 0.1 to 0.5)
    mu = jax.random.uniform(get_random_key(key+8), minval=0.1, maxval=0.5, shape=(s_size,))
    h = jax.random.uniform(get_random_key(key+9), minval=0.1, maxval=0.5, shape=(s_size,))
    dataset = jnp.stack([x, y, a, b, c, d, e, f, mu, h], axis=1)
    dataset = dataset.repeat(steps, axis=0)
    t_eval_set = jnp.tile(t_eval, reps=s_size).reshape(-1, 1)
    dataset = jnp.concatenate([t_eval_set, dataset], axis=1)
    return dataset

def rel_l2_error(pred, true):
    """ Compute the relative L2 error between the predicted and true values """
    return jnp.sqrt(jnp.sum((pred - true)**2, axis=0)) / jnp.sqrt(jnp.sum(true**2, axis=0))
    # return jnp.linalg.norm(true - pred, 2) / jnp.linalg.norm(true, 2)


def get_batch_data(dataset, steps, t_eval, training_batch_size):
    """ Get a randomly shuffled batch data for training the model """
    train_init_indices = jnp.where(dataset[:, 0] == 0)[0] # Select the indices where the time is 0
    train_init_indices = np.random.choice(train_init_indices, size=training_batch_size, replace=False) # Randomly select init training indices half the size of the batch

    random_batch_of_features = dataset[train_init_indices, :]
    random_batch_of_features = random_batch_of_features.repeat(steps, axis=0)
    t_eval = jnp.tile(t_eval, reps=training_batch_size)
    random_batch_of_features = random_batch_of_features.at[:, 0].set(t_eval)
    
    return random_batch_of_features

def return_nnmodel(hp, ran_key, num_features, num_targets, num_layers, num_neurons_per_layer):
    ## Get the network layer sizes and initialize the network parameters
    sizes = hp.get_network_layer_sizes(num_features, num_targets, num_layers, num_neurons_per_layer)
    print(sizes)
    net_init_key, ran_key = jax.random.split(ran_key) # This just returns a new random key
    init_params = hp.get_init_network_params(sizes, net_init_key) # Calls utils function to initialize the network parameters by passing sizes and random key

    ## Define the 1D function that we want the model to learn and run it once so the model is JIT-compiled
    ran_key, func_key = jax.random.split(ran_key)
    random_feature_array = jax.random.uniform(func_key, minval=-1, maxval=1, shape=(num_features, ))
    _ = feedforward_prediction(init_params, random_feature_array, silu)
    params = deepcopy(init_params)

    return params

def return_func_output(eqn_num, state, func):
    # For 8D coupled damped oscillator: x, y, a, b, c, d, e, f, mu, h
    return jax.vmap(lambda x, y, a, b, c, d, e_, f_, mu, h: func[eqn_num](x, y, a, b, c, d, e_, f_, mu, h))(
        state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4], state[:, 5], state[:, 6], state[:, 7], state[:, 8], state[:, 9])

def get_intr_time_coupled_sums(activations, steps, t, t2, t3):
    ## For x state_var
    term_1x = activations[:, 1]

    intr_2x = return_func_output(eqn_num=0, state=activations[::steps, 1:], func=ft_funcs)
    intr_2x = intr_2x.repeat(steps, axis=0)
    term_2x = t * intr_2x 

    intr_3x = return_func_output(eqn_num=0, state=activations[::steps, 1:], func=st_funcs)
    intr_3x = intr_3x.repeat(steps, axis=0)
    term_3x = t2 * intr_3x

    intr_4x = return_func_output(eqn_num=0, state=activations[::steps, 1:], func=tt_funcs)
    intr_4x = intr_4x.repeat(steps, axis=0)
    term_4x = t3 * intr_4x
    
    ps_tc_x = term_1x + term_2x + term_3x + term_4x

    ## For y state_var
    term_1y = activations[:, 2] 
    
    intr_2y = return_func_output(eqn_num=1, state=activations[::steps, 1:], func=ft_funcs)
    intr_2y = intr_2y.repeat(steps, axis=0)
    term_2y = t * intr_2y 
    
    intr_3y = return_func_output(eqn_num=1, state=activations[::steps, 1:], func=st_funcs)
    intr_3y = intr_3y.repeat(steps, axis=0)
    term_3y = t2 * intr_3y

    intr_4y = return_func_output(eqn_num=1, state=activations[::steps, 1:], func=tt_funcs)
    intr_4y = intr_4y.repeat(steps, axis=0)
    term_4y = t3 * intr_4y

    ps_tc_y = term_1y + term_2y + term_3y + term_4y

    ## For z state_var
    term_1z = activations[:, 3]

    intr_2z = return_func_output(eqn_num=2, state=activations[::steps, 1:], func=ft_funcs)
    intr_2z = intr_2z.repeat(steps, axis=0)
    term_2z = t * intr_2z

    intr_3z = return_func_output(eqn_num=2, state=activations[::steps, 1:], func=st_funcs)
    intr_3z = intr_3z.repeat(steps, axis=0)
    term_3z = t2 * intr_3z

    intr_4z = return_func_output(eqn_num=2, state=activations[::steps, 1:], func=tt_funcs)
    intr_4z = intr_4z.repeat(steps, axis=0)
    term_4z = t3 * intr_4z

    ps_tc_z = term_1z + term_2z + term_3z + term_4z

    ## For a state_var
    term_1a = activations[:, 4]

    intr_2a = return_func_output(eqn_num=3, state=activations[::steps, 1:], func=ft_funcs)
    intr_2a = intr_2a.repeat(steps, axis=0)
    term_2a = t * intr_2a

    intr_3a = return_func_output(eqn_num=3, state=activations[::steps, 1:], func=st_funcs)
    intr_3a = intr_3a.repeat(steps, axis=0)
    term_3a = t2 * intr_3a

    intr_4a = return_func_output(eqn_num=3, state=activations[::steps, 1:], func=tt_funcs)
    intr_4a = intr_4a.repeat(steps, axis=0)
    term_4a = t3 * intr_4a

    ps_tc_a = term_1a + term_2a + term_3a + term_4a

    ## Compute partial terms for the loss function
    lps_tc_x = intr_2x + t * intr_3x + t2 * intr_4x
    lps_tc_y = intr_2y + t * intr_3y + t2 * intr_4y
    lps_tc_z = intr_2z + t * intr_3z + t2 * intr_4z
    lps_tc_a = intr_2a + t * intr_3a + t2 * intr_4a

    return ps_tc_x, ps_tc_y, ps_tc_z, ps_tc_a, lps_tc_x, lps_tc_y, lps_tc_z, lps_tc_a

def get_time_terms(batch_size, t_eval):
    train_t = jnp.tile(t_eval, reps=batch_size)
    train_t2 = (train_t**2)/2
    train_t3 = (train_t**3)/6
    train_t4 = (train_t**4)/24

    return train_t, train_t2, train_t3, train_t4

@click.command()
# @click.option("--t", default=2, help="Time horizon for the forward prediction")
@click.option("--plotpred", default=False, help="Plot prediction dataset")
@click.option("--savepred", default=False, help="Save prediction dataset") 
@click.option("--savemetrics", default=False, help="Save metrics from the training process") 
@click.option("--name", default="jModel_test", help="Name of the model")
def create(plotpred: bool, savepred: bool, savemetrics: bool, name: str):
    """ Main function that computes the training and prediction of the neural network model """
    # SEED = np.random.randint(0, 100_000)
    SEED = 9020
    plot_dataset = False
    plot_prediction = plotpred
    save_predictions = savepred
    save_metrics = savemetrics
    save_name = name
    os.makedirs(f"preds/{save_name}", exist_ok=True)
    os.makedirs(f"figs/{save_name}", exist_ok=True)

    ran_key = jax.random.PRNGKey(SEED)
    num_features, num_targets = 11, 1  # 1 time + 8 state + 2 param
    num_layers, num_neurons_per_layer = 1, 64

    global batched_prediction
    batched_prediction = jax.vmap(feedforward_prediction, in_axes=(None, 0, None))

    hp = Helpers()
    # 8 models for 8 state variables
    model_list = []
    for _ in range(8):
        model_list.append(return_nnmodel(hp, ran_key, num_features, num_targets, num_layers, num_neurons_per_layer))
    model = model_list

    ## Create the dataset from random samples in the range specified
    for iter in range(1):
        duration = 3
        time_interval = 0.1
        number_of_samples = 100
        print(f"Duration: {duration}, Time Interval: {time_interval}")

        steps = int(duration / time_interval)
        t_eval = jnp.linspace(0, duration, num=int(steps))

        ## Create the training dataset for the given duration - This consists of only the initial conditions of the model
        dataset = sample_generation(number_of_samples, steps, t_eval, key=9020)
        dataset = jnp.array(dataset)
        print(f"Train Dataset created with shape: \t{dataset.shape}")

        ## Create the validation dataset for the given duration - This consists of only the initial conditions of the model
        validation_dataset = sample_generation(int(number_of_samples/2), steps, t_eval, key=1002)
        validation_dataset = jnp.array(validation_dataset)
        print(f"Validation Dataset created with shape: \t{validation_dataset.shape}")
        
        adam_state = initialize_adam_state(model, num_var=6)  # Initialize Adam state
        batch_counter = 0
        epochs = 500
        lr = 1e-3
        training_batch_size = 24
        validation_batch_size = 12

        global ft_funcs, st_funcs, tt_funcs
        ft_funcs, st_funcs, tt_funcs = get_ode_functions()

        train_t, train_t2, train_t3, train_t4 = get_time_terms(training_batch_size, t_eval)
        val_t, val_t2, val_t3, val_t4 = get_time_terms(validation_batch_size, t_eval)

        logs = ""
        log_entry = (f"SEED: {SEED}\nEpochs: {epochs}\nLearning Rate: {lr}\nDuration: {duration}\nTime Interval: {time_interval}\nSteps: {steps}\nTraining Batch Size: {training_batch_size}\nValidation Batch Size: {validation_batch_size}\nNumber of Unique Training Points: {number_of_samples}\nTraining Dataset Shape: {dataset.shape}\nValidation Dataset Shape: {validation_dataset.shape}\n\n")
        logs += log_entry
        
        init_time = time()

        for itrain in range(batch_counter, batch_counter+epochs):
            activations = get_batch_data(dataset, steps, t_eval, training_batch_size) # Time derived dataset

            # You will need to update get_intr_time_coupled_sums and adam_update_gd for 6D if you want full Taylor expansion support
            # For now, you can adapt the training loop to use the new model and ODE system as above
            # (This is a placeholder for the actual 6D Taylor expansion logic)
            pass  # TODO: Implement 6D Taylor expansion and training logic

            # Validation loop every 500 batches, where we do the same as above but on the validation dataset
            if itrain % 100 == 0:
                validation_activations = get_batch_data(validation_dataset, steps, t_eval, validation_batch_size) # validation_batch_size*shape x 4
                print(validation_activations.shape)
                validation_target = define_ode(validation_activations[::steps], t_eval, duration) # validation_batch_size*shape x 2

                ps_tc_x, ps_tc_y, ps_tc_z, ps_tc_a, _, _, _, _ = get_intr_time_coupled_sums(validation_activations, steps, val_t, val_t2, val_t3)
                validation_prediction_x = jnp.ravel(batched_prediction(model[0], validation_activations, silu)) # validation_batch_size*shape x 1
                validation_prediction_y = jnp.ravel(batched_prediction(model[1], validation_activations, silu)) # validation_batch_size*shape x 1
                validation_prediction_z = jnp.ravel(batched_prediction(model[2], validation_activations, silu)) # validation_batch_size*shape x 1
                validation_prediction_a = jnp.ravel(batched_prediction(model[3], validation_activations, silu)) # validation_batch_size*shape x 1

                time_coupled_x = ps_tc_x + val_t4 * validation_prediction_x
                time_coupled_y = ps_tc_y + val_t4 * validation_prediction_y
                time_coupled_z = ps_tc_z + val_t4 * validation_prediction_z
                time_coupled_a = ps_tc_a + val_t4 * validation_prediction_a
                
                mse_lossx = mse_loss(time_coupled_x, validation_target[:, 0]) 
                mse_lossy = mse_loss(time_coupled_y, validation_target[:, 1])
                mse_lossz = mse_loss(time_coupled_z, validation_target[:, 2])
                mse_lossa = mse_loss(time_coupled_a, validation_target[:, 3])
                mae_lossx = mae_loss(time_coupled_x, validation_target[:, 0])
                mae_lossy = mae_loss(time_coupled_y, validation_target[:, 1])
                mae_lossz = mae_loss(time_coupled_z, validation_target[:, 2])
                mae_lossa = mae_loss(time_coupled_a, validation_target[:, 3])
                rmse_lossx = rmse_loss(time_coupled_x, validation_target[:, 0])
                rmse_lossy = rmse_loss(time_coupled_y, validation_target[:, 1])
                rmse_lossz = rmse_loss(time_coupled_z, validation_target[:, 2])
                rmse_lossa = rmse_loss(time_coupled_a, validation_target[:, 3])
                print(f"Batch {itrain}:\nValidation -> MSE: {((mse_lossx + mse_lossy + mse_lossz + mse_lossa)/4):.5f}\tMAE: {((mae_lossx + mae_lossy + mae_lossz + mae_lossa)/4):.5f}\tRMSE: {((rmse_lossx + rmse_lossy + rmse_lossz + rmse_lossa)/4):.5f}")

                log_entry = (f"\nBatch {itrain}:\nMSE = {mse_lossx:.5f}, {mse_lossy:.5f}, {mse_lossz:.5f}, {mse_lossa:.5f}\tMAE = {mae_lossx:.5f}, {mae_lossy:.5f}, {mae_lossz:.5f}, {mae_lossa:.5f}\tRMSE = {rmse_lossx:.5f}, {rmse_lossy:.5f}, {rmse_lossz:.5f} {rmse_lossa:.5f}")
                logs += log_entry
        
        end_time = time()
        
        print(f"Time taken for training: {(end_time - init_time)/60:.2f} mins") 
        logs += f"\n\nTime taken for training: {(end_time - init_time)/60:.2f} mins\n"

    ## Create new training dataset and get the predictions on them
    duration = duration
    time_interval = time_interval
    total_test_data_size = 100

    print(f"Duration: {duration}, Time Interval: {time_interval}")
    steps = int(duration / time_interval)
    t_eval = jnp.linspace(0, duration, num=int(steps))

    test_t, test_t2, test_t3, test_t4 = get_time_terms(total_test_data_size, t_eval)

    test_dataset = sample_generation(total_test_data_size, steps, t_eval, key=42)
    test_target = define_ode(test_dataset[::steps], t_eval, duration)

    # For 6D system
    # You will need to update get_intr_time_coupled_sums for 6D if you want full Taylor expansion support
    # For now, just use the model predictions directly
    test_predictions = [jnp.ravel(batched_prediction(model[i], test_dataset, silu)) for i in range(8)]
    pred_batch = np.stack(test_predictions, axis=1)

    ae_val = abs(pred_batch - test_target)
    mae_val = np.mean(ae_val, axis=0)
    rel_l2 = rel_l2_error(pred_batch, test_target)
    mse_val = np.mean(ae_val**2, axis=0)
    rmse_val = np.sqrt(mse_val)
    print(f"Prediction batch MAE: {mae_val}, Relative L2 Error: {rel_l2}, RMSE: {rmse_val}, MSE: {mse_val}")

    if save_predictions:
        np.savetxt(f"preds/{save_name}/{save_name}_test_dataset.txt", np.asarray(test_dataset))
        np.savetxt(f"preds/{save_name}/{save_name}_test_target.txt", np.asarray(test_target))
        np.savetxt(f"preds/{save_name}/{save_name}_pred_batch.txt", pred_batch)

    if plot_prediction:
        hp.plot_creations_2d(t_eval, test_dataset, test_target, steps, name=f"{save_name}_tb", no_of_params=2, no_of_svariables=8, no_plot_figures=10, set=save_name, pred_batch=pred_batch)
        hp.plot_error_distributions(pred_batch, test_target, save_name,  name=f"{save_name}_tb", no_of_svariables=8)

    # Get 1sec and 2sec results
    sec = [1, 2]
    for v in sec:
        d = v
        s = int(d / time_interval)
        t = jnp.linspace(0, d, num=int(s))

        indices = []
        for i in range(0, test_dataset.shape[0], t_eval.shape[0]):
            indices.extend(range(i, i + t.shape[0]))
        indices = np.array(indices)

        td = test_dataset[indices]
        pt, tt = pred_batch[indices], test_target[indices]
        print(td.shape, pt.shape, tt.shape)

        aval = abs(pt - tt)
        mval = np.mean(aval, axis=0)
        rl = rel_l2_error(pt, tt)
        msval = np.mean(aval**2, axis=0)
        rmval = np.sqrt(msval)
        print(f"\nPrediction batch for {v} sec with steps {s}\nMAE: {mval}, Relative L2 Error: {rl}, RMSE: {rmval}, MSE: {msval}")
        log_entry = (f"\nTest second {v}:\nMSE = {msval} - avg: {np.mean(list(msval))}\nMAE = {mval} - avg: {np.mean(list(mval))}\nRMSE = {rmval} - avg: {np.mean(list(rmval))}\n")
        logs += log_entry

        if plot_prediction:
            hp.plot_creations_2d(t[:s], td, tt, s, name=f"{save_name}_tb" + "_" + str(v) + "sec", no_of_params=2, no_of_svariables=8, no_plot_figures=10, set=save_name, pred_batch=pt)
            hp.plot_error_distributions(pt, tt, save_name,  name=f"{save_name}_tb" + "_" + str(v) + "sec", no_of_svariables=8)

    if save_metrics:
        with open(f"preds/{save_name}/{save_name}_train_res.txt", "w") as f:
            f.write("Training Results\n")
            f.write(logs)
            f.write(f"\n\nTest Results\nDuration: {duration}\nTime interval: {time_interval}\nNumber of unique testing points: {total_test_data_size}\nTesting dataset shape: {test_dataset.shape}\n\nTest dataset MAE: {mae_val}\tRelative L2 Error: {rel_l2}\tMSE: {mse_val}\t RMSE: {rmse_val}\n")
            f.write(f"Test Dataset Metrics (per state variable):\n")
            f.write(f"MAE: {mae_val} - avg: {np.mean(list(mae_val))}\n")
            f.write(f"RMSE: {rmse_val} - avg: {np.mean(list(rmse_val))}\n")
            f.write(f"MSE: {mse_val} - avg: {np.mean(list(mse_val))}\n")

if __name__ == "__main__":
    create()
