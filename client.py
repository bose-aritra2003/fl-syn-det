# Client Code
import config
import argparse
import flwr as fl
import tensorflow as tf
from utils import load_dataset, get_model

server_address = "10.24.41.216:5050" # Replace with address where the server is running

# Define Flower client
class FLClient(fl.client.NumPyClient):
    def __init__(self, model, client_id, train_images, train_labels, test_images, test_labels):
        self.model = model
        self.client_id = client_id
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels

    def get_parameters(self, config):
        """Get parameters of the local model."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        print(f"[Client {self.client_id}] Training...")
        self.model.set_weights(parameters)
        self.model.fit(
            self.train_images, 
            self.train_labels, 
            epochs=5, 
            batch_size=32, 
            verbose=1
        )
        return self.model.get_weights(), len(self.train_images), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.client_id}] Evaluating on test dataset...")
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.test_images, self.test_labels, verbose=0)
        print(f"[Client {self.client_id}] Test Accuracy: {accuracy}")
        return loss, len(self.test_images), {"accuracy": accuracy}


def parse_args():
    """Parses command-line arguments for the client."""
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument(
        "--client_id", 
        type=int, 
        required=True, 
        help="Client ID (e.g., 1, 2)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    client_id = args.client_id

    train_path = config.CLIENT_TRAIN_PATH_TEMPLATE.format(client_id=client_id)
    test_path = config.CLIENT_TEST_PATH_TEMPLATE.format(client_id=client_id)

    # Load data
    train_images, train_labels = load_dataset(train_path)
    test_images, test_labels = load_dataset(test_path)    

    # Load and compile model
    model = get_model()
    model.summary()
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )    

    client = FLClient(model, client_id, train_images, train_labels, test_images, test_labels)
    fl.client.start_numpy_client(server_address=server_address, client=client)


if __name__ == "__main__":
    main()