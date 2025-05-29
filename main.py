import os
import jax
import jax.numpy as jnp
import numpy as np
from time import time
import sympy as sp
import click

# Import functions from the new files
from src.training import adam_update_gd, initialize_adam_state, get_intr_time_coupled_sums, batched_prediction, mse_loss, mae_loss, rmse_loss
from src.utils import Helpers
from src.system_definition import define_ode_target, create_dynamical_system, define_ode_system, sample_generation, get_time_terms, get_batch_data
from src.neural_network import return_nnmodel

os.environ["ENABLE_PJRT_COMPATIBILITY"] = "1"

@click.command()
@click.option("--symbols", default="x,y,delta", help="Comma-separated list of symbols (state variables first, then parameters)")
@click.option("--equations", default="y,x - x**3 - delta*y", help="Comma-separated list of ODE equations")
@click.option("--initial_conditions_range", default="-0.5,0.5,-0.5,0.5", help="Comma-separated list of initial conditions")
@click.option("--parameter_ranges", default="0.1,0.5", help="Comma-separated list of parameters")
@click.option("--num_state_vars", default=2, help="Number of state variables in the system rest will be parameters (state variables first, then parameters)")
@click.option("--duration", default=3.0, type=float, help="Duration of the simulation")
@click.option("--time_interval", default=0.1, type=float, help="Time interval for evaluation")
@click.option("--num_samples", default=100, type=int, help="Number of unique initial conditions for training")
@click.option("--training_batch_size", default=24, type=int, help="Batch size for training")
@click.option("--validation_batch_size", default=12, type=int, help="Batch size for validation")
@click.option("--num_train_epochs", default=500, type=int, help="Number of training epochs")
@click.option("--learning_rate", default=5e-2, type=float, help="Learning rate for the Adam optimizer")
@click.option("--num_layers", default=1, type=int, help="Number of hidden layers in the neural network")
@click.option("--num_neurons", default=64, type=int, help="Number of neurons per hidden layer")
@click.option("--keyadd", default="0,99,9", help="List of integers to add to create a unique SEED value")
@click.option("--plotpred", default=False, is_flag=True, help="Plot prediction dataset")
@click.option("--savepred", default=False, is_flag=True, help="Save prediction dataset")
@click.option("--savemetrics", default=False, is_flag=True, help="Save metrics from the training process")
@click.option("--name", default="jModel_parametric", help="Name of the model and output files")
def main(symbols, num_state_vars, equations, duration, time_interval, num_samples, num_train_epochs, learning_rate, training_batch_size, validation_batch_size, num_layers, num_neurons, initial_conditions_range, parameter_ranges, keyadd, plotpred, savepred, savemetrics, name):
    """
    Main function to set up and run the neural network training for a parametric dynamical system.
    """
    print(name)
    keyadd = keyadd.split(",")
    keyadd = [int(k) for k in keyadd]
    print(keyadd)
    # Parse inputs
    symbol_list = [sp.symbols(s.strip()) for s in symbols.split(',')]
    equation_list = [sp.sympify(eq.strip()) for eq in equations.split(',')]

    state_vars_symbols = symbol_list[:num_state_vars]
    params_symbols = symbol_list[num_state_vars:]

    print(equation_list, symbol_list)

    num_features = 1 + num_state_vars + len(params_symbols) # Time + state variables + parameters
    num_targets = 1 # Each neural network predicts a single state variable's contribution

    init_cond_ranges_list = initial_conditions_range.split(',')
    if len(init_cond_ranges_list) != len(state_vars_symbols)*2:
        raise ValueError(f"Number of initial condition ranges ({len(init_cond_ranges_list)}) must match 2 * number of state variables ({num_state_vars * 2})")
    init_cond_ranges = [(float(init_cond_ranges_list[i]), float(init_cond_ranges_list[i+1])) for i in range(0, len(init_cond_ranges_list), 2)]

    param_ranges_list = parameter_ranges.split(',')
    if len(param_ranges_list) != len(params_symbols)*2:
            raise ValueError(f"Number of parameter ranges ({len(param_ranges_list)}) must match 2 * number of parameters ({len(params_symbols) * 2}).")
    param_ranges = [(float(param_ranges_list[i]), float(param_ranges_list[i+1])) for i in range(0, len(param_ranges_list), 2)]

    print(init_cond_ranges, param_ranges)

    # Set up directories
    os.makedirs(f"preds/{name}", exist_ok=True)
    os.makedirs(f"figs/{name}", exist_ok=True)

    # System definition
    ft_funcs, st_funcs, tt_funcs, fot_funcs, system_args = create_dynamical_system(symbol_list, equation_list, state_vars_symbols)
    ode_system_func = define_ode_system(symbol_list, equation_list, state_vars_symbols)

    # Model Initialization
    SEED = 9020 # You can make this a parameter too
    ran_key = jax.random.PRNGKey(SEED)
    hp = Helpers()

    # Initialize models for each state variable
    model = []
    for _ in range(num_state_vars):
        ran_key, model_key = jax.random.split(ran_key)
        model.append(return_nnmodel(hp, model_key, num_features, num_targets, num_layers, num_neurons))

    # Data Generation
    steps = int(duration / time_interval)
    t_eval = jnp.linspace(0, duration, num=steps)

    dataset = sample_generation(num_samples, steps, t_eval, init_cond_ranges, param_ranges, key=9020, keyadd=keyadd)
    validation_dataset = sample_generation(int(num_samples/2), steps, t_eval, init_cond_ranges, param_ranges, key=1002, keyadd=keyadd)

    adam_state = initialize_adam_state(model) # Initialize Adam state for the list of models
    training_init_indices_in_batch = jnp.arange(0, training_batch_size * steps, steps)
    validation_init_indices_in_batch = jnp.arange(0, validation_batch_size * steps, steps)

    train_t, train_t2, train_t3, train_t4 = get_time_terms(training_batch_size, t_eval)
    val_t, val_t2, val_t3, val_t4 = get_time_terms(validation_batch_size, t_eval)

    print(train_t.shape, train_t3.shape, val_t3.shape, val_t.shape)

    logs = ""
    log_entry = (f"SEED: {SEED}\nEpochs: {num_train_epochs}\nLearning Rate: {learning_rate}\nDuration: {duration}\nTime Interval: {time_interval}\nSteps: {steps}\nTraining Batch Size: {training_batch_size}\nValidation Batch Size: {validation_batch_size}\nNumber of Unique Training Points: {num_samples}\nTraining Dataset Shape: {dataset.shape}\nValidation Dataset Shape: {validation_dataset.shape}\nSymbols: {symbols}\nEquations: {equations}\nInitial Conditions Range: {initial_conditions_range}\nParameter Ranges: {parameter_ranges}\nNumber of State Variables: {num_state_vars}\nNumber of Parameters: {len(params_symbols)}\n\n")
    logs += log_entry

    init_time = time()

    for itrain in range(num_train_epochs):
        activations = get_batch_data(dataset, steps, t_eval, training_batch_size)
        
        ps_tc, lps_tc = get_intr_time_coupled_sums(activations, steps, train_t, train_t2, train_t3, tuple(ft_funcs), tuple(st_funcs), tuple(tt_funcs), tuple(system_args))
        
        model, adam_state = adam_update_gd(model, adam_state, activations, training_init_indices_in_batch, ps_tc, lps_tc, train_t3, train_t4, learning_rate, tuple(ft_funcs), tuple(fot_funcs), tuple(system_args))
        
        if itrain % 100 == 0:
            validation_activations = get_batch_data(validation_dataset, steps, t_eval, validation_batch_size)
            validation_target_initial_conditions = validation_activations[validation_init_indices_in_batch, 1:] # Shape (validation_batch_size, num_state_vars + num_params)

            validation_target = define_ode_target(validation_target_initial_conditions, t_eval, duration, ode_system_func, params_symbols)
            val_ps_tc, _ = get_intr_time_coupled_sums(validation_activations, steps, val_t, val_t2, val_t3, tuple(ft_funcs), tuple(st_funcs), tuple(tt_funcs), tuple(system_args)) # Pass full system_args

            validation_predictions = jnp.stack([jnp.ravel(batched_prediction(model[i], validation_activations, jax.nn.silu)) for i in range(num_state_vars)], axis=1) # Shape (batch_size, num_state_vars)
            time_coupled_validation_predictions = val_ps_tc + val_t4[:, None] * validation_predictions # Use broadcasting

            mse_losses = jnp.array([mse_loss(time_coupled_validation_predictions[:, i], validation_target[:, i]) for i in range(num_state_vars)])
            mae_losses = jnp.array([mae_loss(time_coupled_validation_predictions[:, i], validation_target[:, i]) for i in range(num_state_vars)])
            rmse_losses = jnp.array([rmse_loss(time_coupled_validation_predictions[:, i], validation_target[:, i]) for i in range(num_state_vars)])

            avg_mse = jnp.mean(mse_losses)
            avg_mae = jnp.mean(mae_losses)
            avg_rmse = jnp.mean(rmse_losses)

            print(f"Epoch {itrain}:\nValidation -> MSE: {avg_mse:.5f}\tMAE: {avg_mae:.5f}\tRMSE: {avg_rmse:.5f}")

            log_entry = (f"\nEpoch {itrain}:\nMSE = {[f'{m:.5f}' for m in mse_losses]}\tMAE = {[f'{m:.5f}' for m in mae_losses]}\tRMSE = {[f'{m:.5f}' for m in rmse_losses]}")
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

    test_dataset = sample_generation(total_test_data_size, steps, t_eval, init_cond_ranges, param_ranges, key=42, keyadd=keyadd)
    test_target = define_ode_target(test_dataset[::steps, 1:], t_eval, duration, ode_system_func, params_symbols)

    test_ps_tc, _ = get_intr_time_coupled_sums(test_dataset, steps, test_t, test_t2, test_t3, tuple(ft_funcs), tuple(st_funcs), tuple(tt_funcs), tuple(system_args)) # Pass full system_args
    test_predictions = jnp.stack([jnp.ravel(batched_prediction(model[i], test_dataset, jax.nn.silu)) for i in range(num_state_vars)], axis=1) # Shape (batch_size, num_state_vars)
    pred_batch = test_ps_tc + test_t4[:, None] * test_predictions # Use broadcasting

    ae_val = abs(pred_batch - test_target)
    mae_val = np.mean(ae_val, axis=0)
    mse_val = np.mean(ae_val**2, axis=0)
    rmse_val = np.sqrt(mse_val)
    print(f"Prediction batch MAE: {mae_val}, RMSE: {rmse_val}, MSE: {mse_val}")
    
    if savepred:
        np.savetxt(f"preds/{name}/{name}_test_dataset.txt", np.asarray(test_dataset))
        np.savetxt(f"preds/{name}/{name}_test_target.txt", np.asarray(test_target))
        np.savetxt(f"preds/{name}/{name}_pred_batch.txt", pred_batch)

    if plotpred:
        # Pass the full test_dataset to the plotting function to access initial conditions for titles
        hp.plot_creations_2d(t_eval, test_dataset, np.array(test_target), steps, name=f"{name}_test", no_of_svariables=num_state_vars, state_vars_symbols=state_vars_symbols, no_of_params=len(params_symbols), no_plot_figures=min(10, total_test_data_size), set=name, pred_batch=pred_batch)
        hp.plot_error_distributions(pred_batch, np.array(test_target), name=f"{name}_test", set=name, no_of_svariables=num_state_vars)

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
        msval = np.mean(aval**2, axis=0)
        rmval = np.sqrt(msval)
        print(f"\nPrediction batch for {v} sec with steps {s}\nMAE: {mval}, RMSE: {rmval}, MSE: {msval}")
        log_entry = (f"\nTest second {v}:\nMSE = {msval} - avg: {np.mean(list(msval))}\nMAE = {mval} - avg: {np.mean(list(mval))}\nRMSE = {rmval} - avg: {np.mean(list(rmval))}\n")
        logs += log_entry

        if plotpred:
            hp.plot_creations_2d(t[:s], td, tt, s, name=f"{name}_tb" + "_" + str(v) + "sec", no_of_params=len(params_symbols), no_of_svariables=num_state_vars, state_vars_symbols=state_vars_symbols, no_plot_figures=10, set=name, pred_batch=pt)
            hp.plot_error_distributions(pt, tt, name,  name=f"{name}_tb" + "_" + str(v) + "sec", no_of_svariables=num_state_vars)

    if savemetrics:
        with open(f"preds/{name}/{name}_train_res.txt", "w") as f:
            f.write("Training Results\n")
            f.write(logs)
            f.write(f"\n\nTest Results\nDuration: {duration}\nTime interval: {time_interval}\nNumber of unique testing points: {total_test_data_size}\nTesting dataset shape: {test_dataset.shape}\nNumber of State Variables: {num_state_vars}\nNumber of Parameters: {len(params_symbols)}\n\n")
            f.write(f"Test Dataset Metrics (per state variable):\n")
            f.write(f"MAE: {mae_val} - avg: {np.mean(list(mae_val))}\n")
            f.write(f"RMSE: {rmse_val} - avg: {np.mean(list(rmse_val))}\n")
            f.write(f"MSE: {mse_val} - avg: {np.mean(list(mse_val))}\n")


if __name__ == "__main__":
    main()
