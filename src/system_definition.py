import sympy as sp
import jax
import jax.numpy as jnp

def create_dynamical_system(symbols, equations, state_vars_symbols, params_symbols):
    """
    Creates symbolic representations and lambdified functions for the dynamical system.

    Args:
        symbols (list): A list of SymPy symbols for state variables and parameters.
        equations (list): A list of SymPy expressions representing the system's ODEs.
    Returns:
        tuple: A tuple containing:
            - funcs (list): List of lambdified functions for the original ODEs.
            - deriv_funcs (list): List of lambdified functions for the first time derivatives.
            - dderiv_funcs (list): List of lambdified functions for the second time derivatives.
            - ddderiv_funcs (list): List of lambdified functions for the third time derivatives.
            - args (list): List of symbols in the order they should be passed to the lambdified functions.
    """
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

    print(F_dot, F_ddot, F_dddot, F_ddddot)

    return funcs, deriv_funcs, dderiv_funcs, ddderiv_funcs, args

def define_ode_system(symbols, equations, state_vars_symbols, params_symbols):
    """
    Defines the ODE system for numerical solving using scipy.integrate.

    Args:
        symbols (list): A list of SymPy symbols for state variables and parameters.
        equations (list): A list of SymPy expressions representing the system's ODEs.
    Returns:
        function: A function that can be used with scipy.integrate.solve_ivp.
    """
    num_state_vars = len(state_vars_symbols)
    state_vars = symbols[:num_state_vars]
    params = symbols[num_state_vars:]
    args = state_vars + params
    assert len(state_vars) + len(params) == len(args)

    diffEq_func = [sp.lambdify(args, e) for e in equations]

    def Eqn(t, y, *params_vals):
        # y contains the state variable values
        # params_vals contains the parameter values
        all_args = list(y) + list(params_vals)
        dydt = [diffEq_func[i](*all_args) for i in range(len(diffEq_func))]
        return dydt

    return Eqn

def return_func_output(eqn_num, state, func, args):
    """
    Computes the output of a lambdified function for a given state.

    Args:
        eqn_num (int): The index of the function to use.
        state (jnp.ndarray): The input state for the function, shape (batch_size, num_state_vars + num_params).
        func (list): List of lambdified functions.
        args (list): List of symbols representing the order of state variables and parameters.
    Returns:
        jnp.ndarray: The output of the function, shape (batch_size,).
    """
    # Ensure the state dimensions match the number of arguments expected by the function
    if state.shape[1] != len(args):
         raise ValueError(f"State dimensions ({state.shape[1]}) do not match the number of arguments ({len(args)}) for the function.")

    vmap_args = tuple(state[:, i] for i in range(state.shape[1]))

    return jax.vmap(lambda *vmap_args: func[eqn_num](*vmap_args))(*vmap_args)