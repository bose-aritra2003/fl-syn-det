# Server Code
import os
import config
import flwr as fl
from glob import glob
import tensorflow as tf
from utils import load_dataset
from typing import Dict, Optional, Tuple
from models.EfficientNetB0Pretrained import EfficientNetB0Pretrained
from strategies.FedAvgWithCheckpointsAndResultsJSON import FedAvgWithCheckpointsAndResultsJSON


def main():
    model = EfficientNetB0Pretrained()

    # Check for existing checkpoints
    checkpoint_dir = "checkpoints"
    checkpoint_files = sorted(glob(os.path.join(checkpoint_dir, "global_model_round_*.keras")))

    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]  # Get the most recent checkpoint
        print(f"Loading model weights from: {latest_checkpoint}")
        model = tf.keras.models.load_model(latest_checkpoint)  # Load the model with weights
    else:
        print("No checkpoint found. Initializing fresh model.")    

    model.summary()
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )

    # Initialize model parameters
    ndarrays = model.get_weights()
    parameters = fl.common.ndarrays_to_parameters(ndarrays)

    strategy = FedAvgWithCheckpointsAndResultsJSON(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        initial_parameters=parameters,
    )
    fl.server.start_server(
        server_address=config.SERVER_ADDRESS, 
        config=fl.server.ServerConfig(num_rounds=config.NUM_ROUNDS), 
        strategy=strategy
    )


def get_evaluate_fn(model):
    test_images, test_labels = load_dataset(config.SERVER_TEST_PATH)
    print("Test images shape:", test_images.shape)
    print("Test label shape:", test_labels.shape)
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        print(f"[Server] Evaluating round {server_round}...")
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
        print(f"[Server] Test Accuracy: {accuracy}")
        return loss, {"accuracy": accuracy}

    return evaluate


if __name__ == "__main__":
    main()