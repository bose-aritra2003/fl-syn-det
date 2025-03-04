# Server Code
import config
import flwr as fl
import tensorflow as tf
from utils import load_dataset
from typing import Dict, Optional, Tuple
from models.EfficientNetB0Pretrained import EfficientNetB0Pretrained


def main():
    model = EfficientNetB0Pretrained()
    model.summary()
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )

    # Initialize model parameters
    ndarrays = model.get_weights()
    parameters = fl.common.ndarrays_to_parameters(ndarrays)

    strategy = fl.server.strategy.FedAvg(
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