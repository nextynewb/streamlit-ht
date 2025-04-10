import tensorflow as tf
from tensorflow.keras import layers, models
import optuna
from tensorflow.keras.datasets import mnist, cifar10
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import json
import os
import joblib
import logging
import time
import argparse
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import io
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable
import sys
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hyperopt.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("neural_hyperopt")

# Configuration
CONFIGURATIONS = {
    "mnist": {
        "input_shape": (28, 28, 1),
        "num_classes": 10,
    },
    "cifar_10": {
        "input_shape": (32, 32, 3),
        "num_classes": 10,
    },
    "fashion_mnist": {
        "input_shape": (28, 28, 1),
        "num_classes": 10,
    }
}

# Hyperparameter spaces
HYPERPARAMETER_SPACES = {
    "common": {
        "optimizer": ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
        "learning_rate": [1.0, 0.1, 0.01, 0.001, 0.0001],
        "batch_size": [32, 64, 128, 256, 512],
        "epochs": [10, 20, 30, 40, 50],
        "activation": ['relu', 'elu', 'selu', 'tanh']
    },
    "mlp": {
        "hidden_layers": [1, 2, 3],
        "units": [64, 128, 256, 512, 1024],
        "dropout_rate": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    },
    "cnn": {
        "kernel_size1": [3, 4, 5],
        "kernel_size2": [3, 4, 5],
        "kernel_size3": [3, 4, 5],
        "filter_1": [16, 32, 64, 96],
        "filter_2": [32, 64, 96, 128],
        "filter_3": [64, 96, 128, 256],
        "dropout_rate": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    },
    "lenet": {
        "kernel_size1": [3, 5],
        "kernel_size2": [3, 5],
        "filter_1": [6, 16, 32],
        "filter_2": [16, 32, 64],
        "fc1_units": [120, 240],
        "fc2_units": [84, 168]
    }
}

# Optimizer mapping
OPTIMIZER_DICT = {
    'SGD': tf.keras.optimizers.SGD,
    'RMSprop': tf.keras.optimizers.RMSprop,
    'Adagrad': tf.keras.optimizers.Adagrad,
    'Adadelta': tf.keras.optimizers.Adadelta,
    'Adam': tf.keras.optimizers.Adam,
    'Adamax': tf.keras.optimizers.Adamax,
    'Nadam': tf.keras.optimizers.Nadam
}

# Sampling algorithms
SAMPLING_ALGORITHMS = {
    "TPE": optuna.samplers.TPESampler,
    "Random": optuna.samplers.RandomSampler,
    "NSGAII": optuna.samplers.NSGAIISampler,
    "NSGAIII": optuna.samplers.NSGAIIISampler,
}

def limit_gpu_memory():
    """Limit TensorFlow GPU memory consumption"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
        except RuntimeError as e:
            logger.error(e)

def load_data(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess dataset"""
    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
        x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255.0
        
    elif dataset_name == "cifar_10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
    elif dataset_name == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
        x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255.0
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    
    return x_train, y_train, x_test, y_test

def build_mlp(trial, input_shape, num_classes):
    """Build MLP model with trial hyperparameters"""
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    
    # Number of hidden layers
    n_layers = trial.suggest_categorical("hidden_layers", HYPERPARAMETER_SPACES["mlp"]["hidden_layers"])
    activation = trial.suggest_categorical("activation", HYPERPARAMETER_SPACES["common"]["activation"])
    
    for i in range(n_layers):
        units = trial.suggest_categorical(f"units_{i}", HYPERPARAMETER_SPACES["mlp"]["units"])
        model.add(layers.Dense(units, activation=activation))
        
        # Add dropout
        dropout_rate = trial.suggest_categorical(f"dropout_{i}", HYPERPARAMETER_SPACES["mlp"]["dropout_rate"])
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compile model
    optimizer_name = trial.suggest_categorical("optimizer", HYPERPARAMETER_SPACES["common"]["optimizer"])
    learning_rate = trial.suggest_categorical("learning_rate", HYPERPARAMETER_SPACES["common"]["learning_rate"])
    optimizer = OPTIMIZER_DICT[optimizer_name](learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_cnn(trial, input_shape, num_classes):
    """Build CNN model with trial hyperparameters"""
    model = models.Sequential()
    
    # First convolutional block
    filters_1 = trial.suggest_categorical("filter_1", HYPERPARAMETER_SPACES["cnn"]["filter_1"])
    kernel_size1 = trial.suggest_categorical("kernel_size1", HYPERPARAMETER_SPACES["cnn"]["kernel_size1"])
    activation = trial.suggest_categorical("activation", HYPERPARAMETER_SPACES["common"]["activation"])
    
    # Add padding='same' to preserve dimensions
    model.add(layers.Conv2D(filters=filters_1, kernel_size=kernel_size1, 
                           activation=activation, padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Second convolutional block
    filters_2 = trial.suggest_categorical("filter_2", HYPERPARAMETER_SPACES["cnn"]["filter_2"])
    kernel_size2 = trial.suggest_categorical("kernel_size2", HYPERPARAMETER_SPACES["cnn"]["kernel_size2"])
    
    model.add(layers.Conv2D(filters=filters_2, kernel_size=kernel_size2, 
                           activation=activation, padding='same'))
    
    # Calculate current feature map size (approximation)
    # For CIFAR-10: 32x32 -> 16x16 after first pooling
    current_size = input_shape[0] // 2
    
    # Only add second pooling if dimensions allow it
    if current_size >= 2:
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        current_size //= 2
    
    # Third convolutional block (optional)
    if trial.suggest_categorical("add_third_conv", [True, False]):
        filters_3 = trial.suggest_categorical("filter_3", HYPERPARAMETER_SPACES["cnn"]["filter_3"])
        kernel_size3 = trial.suggest_categorical("kernel_size3", HYPERPARAMETER_SPACES["cnn"]["kernel_size3"])
        
        model.add(layers.Conv2D(filters=filters_3, kernel_size=kernel_size3, 
                               activation=activation, padding='same'))
        
        # Only add third pooling if dimensions allow it
        if current_size >= 2:
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten and dense layers
    model.add(layers.Flatten())
    
    # Dropout before dense layer
    dropout_rate = trial.suggest_categorical("dropout_rate", HYPERPARAMETER_SPACES["cnn"]["dropout_rate"])
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compile model
    optimizer_name = trial.suggest_categorical("optimizer", HYPERPARAMETER_SPACES["common"]["optimizer"])
    learning_rate = trial.suggest_categorical("learning_rate", HYPERPARAMETER_SPACES["common"]["learning_rate"])
    optimizer = OPTIMIZER_DICT[optimizer_name](learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_lenet(trial, input_shape, num_classes):
    """Build LeNet model with trial hyperparameters"""
    model = models.Sequential()
    
    # First convolutional layer
    filters_1 = trial.suggest_categorical("filter_1", HYPERPARAMETER_SPACES["lenet"]["filter_1"])
    kernel_size1 = trial.suggest_categorical("kernel_size1", HYPERPARAMETER_SPACES["lenet"]["kernel_size1"])
    activation = trial.suggest_categorical("activation", HYPERPARAMETER_SPACES["common"]["activation"])
    
    model.add(layers.Conv2D(filters=filters_1, kernel_size=kernel_size1, 
                           activation=activation, input_shape=input_shape))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    
    # Second convolutional layer
    filters_2 = trial.suggest_categorical("filter_2", HYPERPARAMETER_SPACES["lenet"]["filter_2"])
    kernel_size2 = trial.suggest_categorical("kernel_size2", HYPERPARAMETER_SPACES["lenet"]["kernel_size2"])
    
    model.add(layers.Conv2D(filters=filters_2, kernel_size=kernel_size2, activation=activation))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    
    # Flatten layer
    model.add(layers.Flatten())
    
    # Fully connected layers
    fc1_units = trial.suggest_categorical("fc1_units", HYPERPARAMETER_SPACES["lenet"]["fc1_units"])
    model.add(layers.Dense(fc1_units, activation=activation))
    
    fc2_units = trial.suggest_categorical("fc2_units", HYPERPARAMETER_SPACES["lenet"]["fc2_units"])
    model.add(layers.Dense(fc2_units, activation=activation))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compile model
    optimizer_name = trial.suggest_categorical("optimizer", HYPERPARAMETER_SPACES["common"]["optimizer"])
    learning_rate = trial.suggest_categorical("learning_rate", HYPERPARAMETER_SPACES["common"]["learning_rate"])
    optimizer = OPTIMIZER_DICT[optimizer_name](learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def get_model_builder(model_name):
    """Get the appropriate model builder function"""
    builders = {
        "mlp": build_mlp,
        "cnn": build_cnn,
        "lenet": build_lenet
    }
    
    if model_name not in builders:
        raise ValueError(f"Unknown model type: {model_name}")
    
    return builders[model_name]

def objective(trial, dataset_name, model_name):
    """Objective function for Optuna optimization"""
    try:
        # Load data
        x_train, y_train, x_test, y_test = load_data(dataset_name)
        
        # Get configuration
        config = CONFIGURATIONS[dataset_name]
        input_shape = config["input_shape"]
        num_classes = config["num_classes"]
        
        # Build model
        model_builder = get_model_builder(model_name)
        model = model_builder(trial, input_shape, num_classes)
        
        # Get hyperparameters for training
        batch_size = trial.suggest_categorical("batch_size", HYPERPARAMETER_SPACES["common"]["batch_size"])
        epochs = trial.suggest_categorical("epochs", HYPERPARAMETER_SPACES["common"]["epochs"])
        
        # Add callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
        
        # Train model
        start_time = time.time()
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        training_time = time.time() - start_time
        
        # Evaluate model
        score = model.evaluate(x_test, y_test, verbose=0)
        accuracy = score[1]
        
        # Calculate F1 score
        y_pred = np.argmax(model.predict(x_test), axis=-1)
        f1 = f1_score(np.argmax(y_test, axis=1), y_pred, average='macro')

        # Count model parameters
        model_params = model.count_params()
        
        # Log results
        logger.info(f"Trial {trial.number}: accuracy={accuracy:.4f}, params={model_params}, f1={f1:.4f}, time={training_time:.2f}s")
        
        return accuracy, model_params, f1
        
    except Exception as e:
        # Log the error
        logger.error(f"Trial {trial.number} failed: {str(e)}")
        
        # Optuna will mark this trial as failed
        raise e

def run_optimization(dataset_name, model_name, algorithm, n_trials=100):
    """Run hyperparameter optimization"""
    # Create output directories if they don't exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Check if a study already exists
    study_path = f"results/{model_name}_{dataset_name}_{algorithm}.pkl"
    if os.path.exists(study_path):
        logger.info(f"Loading existing study from {study_path}")
        study = joblib.load(study_path)
        start_trial = len(study.trials)
        logger.info(f"Resuming from trial {start_trial}")
    else:
        # Create a new study object with the specified algorithm
        sampler = SAMPLING_ALGORITHMS[algorithm]()
        study = optuna.create_study(
            directions=["maximize", "minimize", "maximize"],  # Accuracy, Params, F1
            sampler=sampler,
            study_name=f"{model_name}_{dataset_name}_{algorithm}"
        )
        start_trial = 0
        logger.info(f"Created new study: {model_name}_{dataset_name}_{algorithm}")
    
    # Run optimization
    logger.info(f"Starting optimization for {model_name} on {dataset_name} using {algorithm}")
    
    # Create a progress bar for CLI
    pbar = tqdm(total=n_trials, initial=0, desc="Optimization Progress")
    
    # Create a callback to update progress
    last_trial = start_trial
    
    def progress_callback(study, trial):
        nonlocal last_trial
        current_trial = len(study.trials)
        if current_trial > last_trial:
            pbar.update(current_trial - last_trial)
            last_trial = current_trial
            
            # Update description with latest metrics
            if trial.values:
                accuracy, params, f1 = trial.values
                pbar.set_description(f"Trial {current_trial}: Acc={accuracy:.4f}, F1={f1:.4f}")
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, dataset_name, model_name),
        n_trials=n_trials,
        gc_after_trial=True,
        callbacks=[progress_callback]
    )
    
    # Close progress bar
    pbar.close()
    
    # Save study
    joblib.dump(study, study_path)
    
    # Extract results
    trials = study.get_trials(deepcopy=False)
    results = []
    
    for trial in trials:
        if trial.values is None:
            continue
            
        accuracy, params, f1 = trial.values
        results.append({
            "trial": trial.number,
            "accuracy": accuracy,
            "params": params,
            "f1": f1,
            "hyperparameters": trial.params
        })
    
    # Save results to JSON
    with open(f"results/{model_name}_{dataset_name}_{algorithm}.json", "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Optimization completed. Results saved to results/{model_name}_{dataset_name}_{algorithm}.json")
    
    return study, results, start_trial

def run_optimization_with_progress(dataset_name, model_name, algorithm, n_trials, progress_bar, status_text, start_trial=0):
    """Run hyperparameter optimization with progress updates for Streamlit"""
    # Create output directories if they don't exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Check if a study already exists
    study_path = f"results/{model_name}_{dataset_name}_{algorithm}.pkl"
    if os.path.exists(study_path) and start_trial > 0:
        study = joblib.load(study_path)
    else:
        # Create a new study object with the specified algorithm
        sampler = SAMPLING_ALGORITHMS[algorithm]()
        study = optuna.create_study(
            directions=["maximize", "minimize", "maximize"],  # Accuracy, Params, F1
            sampler=sampler,
            study_name=f"{model_name}_{dataset_name}_{algorithm}"
        )
    
    # Create a callback to update progress
    last_trial = start_trial
    
    def progress_callback(study, trial):
        nonlocal last_trial
        current_trial = len(study.trials)
        if current_trial > last_trial:
            last_trial = current_trial
            # Update progress bar
            progress = min(1.0, (current_trial - start_trial) / n_trials)
            progress_bar.progress(progress)
            
            # Update status text
            if trial.values:
                accuracy, params, f1 = trial.values
                status_text.text(f"Trial {current_trial}/{start_trial + n_trials}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
            else:
                status_text.text(f"Trial {current_trial}/{start_trial + n_trials}: Failed")
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, dataset_name, model_name),
        n_trials=n_trials,
        gc_after_trial=True,
        callbacks=[progress_callback]
    )
    
    # Save study
    joblib.dump(study, study_path)
    
    # Extract results
    trials = study.get_trials(deepcopy=False)
    results = []
    
    for trial in trials:
        if trial.values is None:
            continue
            
        accuracy, params, f1 = trial.values
        results.append({
            "trial": trial.number,
            "accuracy": accuracy,
            "params": params,
            "f1": f1,
            "hyperparameters": trial.params
        })
    
    # Save results to JSON
    with open(f"results/{model_name}_{dataset_name}_{algorithm}.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # Complete the progress bar
    progress_bar.progress(1.0)
    status_text.text(f"Optimization completed with {len(study.trials)} trials!")
    
    return study, results, start_trial

def visualize_results(study, title="Optimization Results"):
    """Visualize the Pareto front in 3D"""
    # Extract values from trials
    trials = study.get_trials(deepcopy=False)
    values = [trial.values for trial in trials if trial.values is not None]
    
    if not values:
        return None
    
    # Extract individual objectives
    accuracies = [accuracy for accuracy, _, _ in values]
    params = [param for _, param, _ in values]
    f1s = [f1 for _, _, f1 in values]
    
    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(accuracies, params, f1s, c=params, cmap='viridis', alpha=0.8)
    
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Parameters')
    ax.set_zlabel('F1 Score')
    ax.set_title(title)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Number of Parameters')
    
    # Save figure to bytes for Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    plt.close(fig)
    return buf

def plot_trial_history(study):
    """Plot the history of trial accuracies"""
    trials = study.get_trials(deepcopy=False)
    valid_trials = [t for t in trials if t.values is not None]
    
    if not valid_trials:
        return None
    
    # Extract trial numbers and accuracies
    trial_numbers = [t.number for t in valid_trials]
    accuracies = [t.values[0] for t in valid_trials]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(trial_numbers, accuracies, 'o-', alpha=0.7)
    
    # Add moving average
    window = min(10, len(accuracies))
    if window > 1:
        moving_avg = np.convolve(accuracies, np.ones(window)/window, mode='valid')
        ax.plot(trial_numbers[window-1:], moving_avg, 'r-', linewidth=2, 
                label=f'{window}-trial Moving Average')
    
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Accuracy')
    ax.set_title('Trial Accuracy History')
    ax.grid(True, alpha=0.3)
    
    if window > 1:
        ax.legend()
    
    # Save figure to bytes for Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    plt.close(fig)
    return buf

def get_best_models(study, top_n=5):
    """Get the top N models based on accuracy"""
    trials = study.get_trials(deepcopy=False)
    
    # Filter out trials with no values
    valid_trials = [t for t in trials if t.values is not None]
    
    # Sort by accuracy (first objective)
    sorted_trials = sorted(valid_trials, key=lambda t: t.values[0], reverse=True)
    
    return sorted_trials[:top_n]

def train_best_model(dataset_name, model_name, params):
    """Train the best model with the given parameters"""
    # Load data
    x_train, y_train, x_test, y_test = load_data(dataset_name)
    
    # Get configuration
    config = CONFIGURATIONS[dataset_name]
    input_shape = config["input_shape"]
    num_classes = config["num_classes"]
    
    # Create a fixed trial with the best parameters
    class FixedTrial:
        def __init__(self, params):
            self.params = params
            
        def suggest_categorical(self, name, choices):
            if name in self.params:
                return self.params[name]
            # Default to first choice if parameter not found
            return choices[0]
    
    # Build model
    model_builder = get_model_builder(model_name)
    model = model_builder(FixedTrial(params), input_shape, num_classes)
    
    # Train model
    batch_size = params.get("batch_size", 64)
    epochs = params.get("epochs", 20)
    
    # Add callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
    
    # Train model
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    
    # Calculate F1 score
    y_pred = np.argmax(model.predict(x_test), axis=-1)
    f1 = f1_score(np.argmax(y_test, axis=1), y_pred, average='macro')
    
    # Save model
    model.save(f"models/best_{model_name}_{dataset_name}.h5")
    
    return model, history, test_acc, f1

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='lower right')
    
    # Loss plot
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    
    # Save figure to bytes for Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    plt.close(fig)
    return buf

def load_study(dataset_name, model_name, algorithm):
    """Load a saved study if it exists"""
    study_path = f"results/{model_name}_{dataset_name}_{algorithm}.pkl"
    
    if os.path.exists(study_path):
        return joblib.load(study_path)
    
    return None

def streamlit_ui():
    """Streamlit UI for the hyperparameter optimization tool"""
    st.set_page_config(page_title="Neural Network Hyperparameter Optimization", layout="wide")
    
    st.title("Neural Network Hyperparameter Optimization")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    dataset_name = st.sidebar.selectbox(
        "Dataset",
        list(CONFIGURATIONS.keys()),
        index=1  # Default to cifar_10
    )
    
    model_name = st.sidebar.selectbox(
        "Model Architecture",
        ["mlp", "cnn", "lenet"],
        index=1  # Default to CNN
    )
    
    algorithm = st.sidebar.selectbox(
        "Optimization Algorithm",
        list(SAMPLING_ALGORITHMS.keys()),
        index=3  # Default to NSGAIII
    )
    
    n_trials = st.sidebar.slider(
        "Number of Trials",
        min_value=10,
        max_value=200,
        value=50,
        step=10
    )
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["Run Optimization", "View Results", "Train Best Model"])
    
    # Tab 1: Run Optimization
    with tab1:
        st.header("Run Hyperparameter Optimization")
        
        st.write(f"Selected configuration: {model_name.upper()} on {dataset_name} using {algorithm}")
        
        # Check if study already exists
        existing_study = load_study(dataset_name, model_name, algorithm)
        if existing_study:
            start_trial = len(existing_study.trials)
            st.info(f"A previous study with {start_trial} trials exists.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Start New Study"):
                    # Delete existing study
                    if os.path.exists(f"results/{model_name}_{dataset_name}_{algorithm}.json"):
                        os.remove(f"results/{model_name}_{dataset_name}_{algorithm}.json")
                    
                    # Create placeholders for progress updates
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results_area = st.container()
                    
                    with st.spinner(f"Running {n_trials} trials..."):
                        # Limit GPU memory
                        limit_gpu_memory()
                        
                        # Run optimization with progress tracking
                        study, results, _ = run_optimization_with_progress(
                            dataset_name, model_name, algorithm, n_trials, 
                            progress_bar, status_text, 0
                        )
                        
                        # Display results
                        with results_area:
                            st.success(f"Optimization completed with {len(study.trials)} trials!")
                            
                            # Show Pareto front
                            pareto_plot = visualize_results(study, f"Pareto Front: {model_name.upper()} on {dataset_name}")
                            if pareto_plot:
                                st.image(pareto_plot, caption="Pareto Front Visualization")
                            
                            # Show trial history
                            history_plot = plot_trial_history(study)
                            if history_plot:
                                st.image(history_plot, caption="Trial Accuracy History")
                            
                            # Show best trials
                            best_trials = get_best_models(study, top_n=5)
                            
                            if best_trials:
                                st.subheader("Top 5 Models by Accuracy")
                                
                                best_df = pd.DataFrame([
                                    {
                                        "Trial": t.number,
                                        "Accuracy": t.values[0],
                                        "Parameters": t.values[1],
                                        "F1 Score": t.values[2],
                                        **{k: v for k, v in t.params.items() if k in ["optimizer", "learning_rate", "batch_size", "epochs"]}
                                    }
                                    for t in best_trials
                                ])
                                
                                st.dataframe(best_df)
            
            with col2:
                if st.button("Resume Existing Study"):
                    # Create placeholders for progress updates
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results_area = st.container()
                    
                    with st.spinner(f"Running {n_trials} more trials..."):
                        # Limit GPU memory
                        limit_gpu_memory()
                        
                        # Run optimization with progress tracking
                        study, results, start_trial = run_optimization_with_progress(
                            dataset_name, model_name, algorithm, n_trials, 
                            progress_bar, status_text, start_trial
                        )
                        
                        # Display results
                        with results_area:
                            st.success(f"Optimization completed with {len(study.trials)} trials!")
                            
                            # Show Pareto front
                            pareto_plot = visualize_results(study, f"Pareto Front: {model_name.upper()} on {dataset_name}")
                            if pareto_plot:
                                st.image(pareto_plot, caption="Pareto Front Visualization")
                            
                            # Show trial history
                            history_plot = plot_trial_history(study)
                            if history_plot:
                                st.image(history_plot, caption="Trial Accuracy History")
                            
                            # Show best trials
                            best_trials = get_best_models(study, top_n=5)
                            
                            if best_trials:
                                st.subheader("Top 5 Models by Accuracy")
                                
                                best_df = pd.DataFrame([
                                    {
                                        "Trial": t.number,
                                        "Accuracy": t.values[0],
                                        "Parameters": t.values[1],
                                        "F1 Score": t.values[2],
                                        **{k: v for k, v in t.params.items() if k in ["optimizer", "learning_rate", "batch_size", "epochs"]}
                                    }
                                    for t in best_trials
                                ])
                                
                                st.dataframe(best_df)
        else:
            if st.button("Start Optimization"):
                # Create placeholders for progress updates
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_area = st.container()
                
                with st.spinner(f"Running {n_trials} trials..."):
                    # Limit GPU memory
                    limit_gpu_memory()
                    
                    # Run optimization with progress tracking
                    study, results, _ = run_optimization_with_progress(
                        dataset_name, model_name, algorithm, n_trials, 
                        progress_bar, status_text, 0
                    )
                    
                    # Display results
                    with results_area:
                        st.success(f"Optimization completed with {len(study.trials)} trials!")
                        
                        # Show Pareto front
                        pareto_plot = visualize_results(study, f"Pareto Front: {model_name.upper()} on {dataset_name}")
                        if pareto_plot:
                            st.image(pareto_plot, caption="Pareto Front Visualization")
                        
                        # Show trial history
                        history_plot = plot_trial_history(study)
                        if history_plot:
                            st.image(history_plot, caption="Trial Accuracy History")
                        
                        # Show best trials
                        best_trials = get_best_models(study, top_n=5)
                        
                        if best_trials:
                            st.subheader("Top 5 Models by Accuracy")
                            
                            best_df = pd.DataFrame([
                                {
                                    "Trial": t.number,
                                    "Accuracy": t.values[0],
                                    "Parameters": t.values[1],
                                    "F1 Score": t.values[2],
                                    **{k: v for k, v in t.params.items() if k in ["optimizer", "learning_rate", "batch_size", "epochs"]}
                                }
                                for t in best_trials
                            ])
                            
                            st.dataframe(best_df)
    
    # Tab 2: View Results
    with tab2:
        st.header("View Optimization Results")
        
        # Check if results exist
        study = load_study(dataset_name, model_name, algorithm)
        
        if study is None:
            st.warning(f"No results found for {model_name} on {dataset_name} using {algorithm}. Run optimization first.")
        else:
            st.success(f"Found study with {len(study.trials)} trials.")
            
            # Show Pareto front
            pareto_plot = visualize_results(study, f"Pareto Front: {model_name.upper()} on {dataset_name}")
            if pareto_plot:
                st.image(pareto_plot, caption="Pareto Front Visualization")
            
            # Show trial history
            history_plot = plot_trial_history(study)
            if history_plot:
                st.image(history_plot, caption="Trial Accuracy History")
            
            # Show all trials
            valid_trials = [t for t in study.trials if t.values is not None]
            
            if valid_trials:
                st.subheader("All Completed Trials")
                
                all_df = pd.DataFrame([
                    {
                        "Trial": t.number,
                        "Accuracy": t.values[0],
                        "Parameters": t.values[1],
                        "F1 Score": t.values[2],
                        **{k: v for k, v in t.params.items() if k in ["optimizer", "learning_rate", "batch_size", "epochs"]}
                    }
                    for t in valid_trials
                ])
                
                st.dataframe(all_df)
                
                # Download results
                csv = all_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name=f"{model_name}_{dataset_name}_{algorithm}_results.csv",
                    mime="text/csv"
                )
    
    # Tab 3: Train Best Model
    with tab3:
        st.header("Train Best Model")
        
        # Check if results exist
        study = load_study(dataset_name, model_name, algorithm)
        
        if study is None:
            st.warning(f"No results found for {model_name} on {dataset_name} using {algorithm}. Run optimization first.")
        else:
            # Get best trials
            best_trials = get_best_models(study, top_n=5)
            
            if not best_trials:
                st.warning("No valid trials found in the study.")
            else:
                st.subheader("Select a Model to Train")
                
                # Create a selection for the best models
                best_df = pd.DataFrame([
                    {
                        "Trial": t.number,
                        "Accuracy": t.values[0],
                        "Parameters": t.values[1],
                        "F1 Score": t.values[2]
                    }
                    for t in best_trials
                ])
                
                st.dataframe(best_df)
                
                selected_trial = st.selectbox(
                    "Select Trial to Train",
                    [t.number for t in best_trials],
                    format_func=lambda x: f"Trial {x}"
                )
                
                # Find the selected trial
                selected_trial_obj = next((t for t in best_trials if t.number == selected_trial), None)
                
                if selected_trial_obj:
                    st.write("Selected Hyperparameters:")
                    st.json(selected_trial_obj.params)
                    
                    if st.button("Train Model"):
                        with st.spinner("Training model..."):
                            # Limit GPU memory
                            limit_gpu_memory()
                            
                            # Train model
                            model, history, test_acc, f1 = train_best_model(
                                dataset_name, 
                                model_name, 
                                selected_trial_obj.params
                            )
                            
                            # Show results
                            st.success(f"Model trained successfully! Test Accuracy: {test_acc:.4f}, F1 Score: {f1:.4f}")
                            
                            # Show training history
                            history_plot = plot_training_history(history)
                            st.image(history_plot, caption="Training History")
                            
                            # Model summary
                            summary_str = []
                            model.summary(print_fn=lambda x: summary_str.append(x))
                            st.text("\n".join(summary_str))
                            
                            st.info(f"Model saved to models/best_{model_name}_{dataset_name}.h5")


def command_line_interface():
    """Command-line interface for the hyperparameter optimization tool"""
    parser = argparse.ArgumentParser(description="Neural Network Hyperparameter Optimization")
    
    parser.add_argument("--dataset", type=str, default="cifar_10",
                        choices=list(CONFIGURATIONS.keys()),
                        help="Dataset to use")
    
    parser.add_argument("--model", type=str, default="cnn",
                        choices=["mlp", "cnn", "lenet"],
                        help="Model architecture to use")
    
    parser.add_argument("--algorithm", type=str, default="NSGAIII",
                        choices=list(SAMPLING_ALGORITHMS.keys()),
                        help="Optimization algorithm")
    
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of optimization trials")
    
    parser.add_argument("--ui", action="store_true",
                        help="Launch the Streamlit UI")
    
    parser.add_argument("--train-best", action="store_true",
                        help="Train the best model from previous optimization")
    
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing study")
    
    args = parser.parse_args()
    
    # Launch UI if requested
    if args.ui:
        # This will be handled by the main function
        return args
    
    # Limit GPU memory
    limit_gpu_memory()
    
    if args.train_best:
        # Load study and train best model
        study = load_study(args.dataset, args.model, args.algorithm)
        
        if study is None:
            logger.error(f"No results found for {args.model} on {args.dataset} using {args.algorithm}.")
            return
        
        # Get best trial
        best_trials = get_best_models(study, top_n=1)
        
        if not best_trials:
            logger.error("No valid trials found in the study.")
            return
        
        best_trial = best_trials[0]
        logger.info(f"Training best model (Trial {best_trial.number}) with accuracy {best_trial.values[0]:.4f}")
        
        # Train model
        model, history, test_acc, f1 = train_best_model(
            args.dataset, 
            args.model, 
            best_trial.params
        )
        
        logger.info(f"Model trained successfully! Test Accuracy: {test_acc:.4f}, F1 Score: {f1:.4f}")
        logger.info(f"Model saved to models/best_{args.model}_{args.dataset}.h5")
        
    else:
        # Run optimization
        logger.info(f"Starting optimization for {args.model} on {args.dataset} using {args.algorithm} with {args.trials} trials")
        
        # Check if resuming from existing study
        start_trial = 0
        if args.resume:
            study_path = f"results/{args.model}_{args.dataset}_{args.algorithm}.pkl"
            if os.path.exists(study_path):
                study = joblib.load(study_path)
                start_trial = len(study.trials)
                logger.info(f"Resuming from trial {start_trial}")
            else:
                logger.warning(f"No existing study found at {study_path}. Starting new study.")
        
        study, results, _ = run_optimization(args.dataset, args.model, args.algorithm, args.trials)
        
        # Show best results
        best_trials = get_best_models(study, top_n=5)
        
        if best_trials:
            logger.info("Top 5 Models by Accuracy:")
            
            for i, trial in enumerate(best_trials):
                logger.info(f"{i+1}. Trial {trial.number}: Accuracy={trial.values[0]:.4f}, Params={trial.values[1]}, F1={trial.values[2]:.4f}")
    
    return args

def install_dependencies():
    """Install required dependencies if not already installed"""
    try:
        import streamlit
    except ImportError:
        logger.info("Installing streamlit...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    
    try:
        import optuna
    except ImportError:
        logger.info("Installing optuna...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
    
    try:
        import tqdm
    except ImportError:
        logger.info("Installing tqdm...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])

def main():
    """Main entry point"""
    # Create directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Check if running as script or through streamlit
    is_streamlit = 'streamlit' in sys.modules
    
    if is_streamlit:
        # Running through streamlit, launch UI directly
        streamlit_ui()
    else:
        # Running as script, parse command line arguments
        args = command_line_interface()
        
        if args.ui:
            # Install dependencies if needed
            install_dependencies()
            
            # Launch streamlit
            import subprocess
            subprocess.run(["streamlit", "run", __file__])

if __name__ == "__main__":
    main()
