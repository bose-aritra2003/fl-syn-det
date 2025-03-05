# Server Configuration
NUM_ROUNDS = 50  # Number of federated learning rounds
SERVER_ADDRESS = "localhost:5050"

# Dataset Paths
SERVER_TEST_PATH = "dataset/server/test/"
CLIENT_TRAIN_PATH_TEMPLATE = "dataset/client_{client_id}/train/"
CLIENT_TEST_PATH_TEMPLATE = "dataset/client_{client_id}/test/"