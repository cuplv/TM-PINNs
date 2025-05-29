# Taylor-Model Physics Informed Neural Networks (TM-PINNs)
This repository contains the official code implementation of our NeuS 2025 paper titled "Taylor-Model Physics Informed Neural Networks (TM-PINNs) for solving Ordinary Differential Equations" (Paper @ [Proceedings](https://neus-2025.github.io/files/papers/paper_80.pdf)).


The goal of **TM-PINNs**, is to leverage taylor series expansion to improve the accuracy and efficiency of PINNs for systems with parametric uncertainities. TM-PINNs aim to solve parametric ODEs by combining neural networks with physics-based loss functions. This framework written in JAX supports training neural networks to approximate solutions to systems of ODEs.

## Code Structure

The repository is organized as follows:

### Main Files
- **`main.py`**: The entry point for running the training and evaluation of TM-PINNs. It includes command-line options for configuring the training process.

### Source Code (`src/`)
- **`training.py`**: Contains the training loop, loss functions, and optimization logic (e.g., Adam optimizer).
- **`utils.py`**: Includes helper functions for neural network initialization, plotting, and other utilities.
- **`system_definition.py`**: Defines the ODE/PDE systems, symbolic computation of derivatives, and data generation for training.
- **`neural_network.py`**: Implements the feedforward neural network architecture and initialization logic.

### Paper Code (`complete/`)
This folder contains all the original code used to report results in the paper.

## Installation

### Prerequisites
- Python 3.8 or higher
- JAX with GPU support (optional but recommended for faster training)
### Steps
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd TM-PINNs
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Main Script
The `main.py` script is the entry point for training and evaluating TM-PINNs. You can configure the training process using command-line options.

#### Example Command
```bash
python main.py \
    --symbols "x,y,a,b,c,d" \
    --num_state_vars 2 \
    --equations "a*x - b*x*y,-c*y + d*x*y" \
    --duration 3.0 \
    --time_interval 0.1 \
    --num_samples 100 \
    --num_train_epochs 500 \
    --learning_rate 5e-3 \
    --training_batch_size 24 \
    --validation_batch_size 12 \
    --num_layers 1 \
    --num_neurons 64 \
    --initial_conditions_range "0,1,0,1" \
    --parameter_ranges "0.6,1.0,0.2,0.5,0.5,1.0,0.1,0.4" \
    --keyadd "0,99,777,9,7,77" \
    --name lotkavolterra_run
```

Similarily, if we need to run the system for a Rikitake system which is a 3 dimension 2 parameter system, we can change the above bash script as follows:

```bash
python main.py \
  --symbols "x1,x2,x3,x4,x5,x6,V1,V2,V3,V4,V5,V6" \
  --num_state_vars 6 \
  --equations "(V1/(0.5 + 1) - 0.1*x1),((V2*x1)/(0.5 + x1) - 0.1*x2),((V3*x2)/(0.5 + x2) - 0.1*x3),((V4*x3)/(0.5 + x3) - 0.1*x4),((V5*x4)/(0.5 + x4) - 0.1*x5),((V6*x5)/(0.5 + x5) - 0.1*x6)" \
  --duration 2.0 \
  --time_interval 0.1 \
  --num_samples 100 \
  --num_train_epochs 200 \
  --learning_rate 5e-3 \
  --training_batch_size 24 \
  --validation_batch_size 12 \
  --num_layers 1 \
  --num_neurons 64 \
  --initial_conditions_range "0.1,0.5,0.1,0.5,0.1,0.5,0.1,0.5,0.1,0.5,0.1,0.5" \
  --parameter_ranges "0.5,1.0,0.5,1.0,0.5,1.0,0.5,1.0,0.5,1.0,0.5,1.0" \
  --keyadd "0,1,2,3,4,5,6,7,8,9,10,11" \
  --name michaelis_menten_6dtest2 \
  --plotpred \
  --savepred \
  --savemetrics
```

### Command-Line Options
- **`--symbols`**: Comma-separated list of symbols. First list all state variables, then follow with parameters
- **`--equations`**: Comma-separated list of the system of equations.
- **`--initial_conditions_range`**: Range of values for initial conditions (comma-separated).
- **`--parameter_ranges`**: Range of values for parameters (comma-separated).
- **`--num_state_vars`**: Number of state variables in the system (Everything after `num_state_vars` will be treated as params). 
- **`--duration`**: Duration of the simulation.
- **`--time_interval`**: Time interval of the duration mentioned.
- **`--num_samples`**: Number of unique initial conditions for training.
- **`--training_batch_size`**: Batch size for training.
- **`--validation_batch_size`**: Batch size for validation.
- **`--num_train_epochs`**: Number of training epochs.
- **`--learning_rate`**: Learning rate for the Adam optimizer.
- **`--num_layers`**: Number of hidden layers in the neural network.
- **`--num_neurons`**: Number of neurons per hidden layer.
- **`--keyadd`**: List of random integers to create a unique SEED value (list same size as `symbols`).
- **`--plotpred`**: Flag to plot prediction results.
- **`--savepred`**: Flag to save prediction results.
- **`--savemetrics`**: Flag to save training metrics.
- **`--name`**: Name of the model and output files.

## Example Workflow

1. **Define the System**: Specify the ODE system using symbols and equations.
2. **Train the Model**: Run `main.py` with the desired configuration.
3. **Evaluate the Model**: Use the saved predictions and metrics for analysis.
4. **Visualize Results**: Check the plots in the `figs/` directory.

## Key Features

- **Symbolic Differentiation**: Uses SymPy for symbolic computation of derivatives.
- **Neural Network Training**: Implements a feedforward neural network with customizable architecture.
- **Physics-Informed Loss**: Combines data-driven and physics-based loss functions.
- **Visualization**: Generates trajectory plots and error distributions.

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvement, feel free to open an issue or submit a pull request.
(Feel free to ⭐️ the repository if you find it useful!)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
