# Server Code
import os
import config
import flwr as fl
from glob import glob
import tensorflow as tf
from utils import load_dataset, get_model
from typing import Dict, Optional, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score
from strategies.FedAvgWithCheckpointsAndResultsJSON import FedAvgWithCheckpointsAndResultsJSON

server_address = "0.0.0.0:5050" # Do not change if you are running the server on the development machine

def main():
    model = get_model()

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
        server_address=server_address, 
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

        # Get raw predictions
        predictions = model.predict(test_images, verbose=0)
        predicted_classes = (predictions > 0.5).astype(int)  # Convert to binary labels

        # Compute standard metrics
        loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)

        # Compute additional metrics using sklearn
        precision = precision_score(test_labels, predicted_classes, pos_label=1)  # Precision for "fake" (1)
        recall = recall_score(test_labels, predicted_classes, pos_label=1)  # Recall for "fake" (1)
        f1 = f1_score(test_labels, predicted_classes, pos_label=1)  # F1-score for "fake" (1)

        # Compute AUC using TensorFlow's built-in metric
        auc_metric = tf.keras.metrics.AUC()
        auc_metric.update_state(test_labels, predictions)
        auc = auc_metric.result().numpy()

        print(f"[Server] Test Accuracy: {accuracy}, Precision (Fake): {precision}, Recall (Fake): {recall}, F1 (Fake): {f1}, AUC: {auc}")

        return loss, {
            "accuracy": float(accuracy),
            "precision_fake": float(precision),
            "recall_fake": float(recall),
            "f1_score_fake": float(f1),
            "auc": float(auc)
        }

    return evaluate


if __name__ == "__main__":
    main()