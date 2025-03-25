from flwr.common import FitRes, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedProx

import tensorflow as tf
import json
import os

from utils import get_model

class FedProxWithCheckpointsAndResultsJSON(FedProx):
    """A strategy that keeps the core functionality of FedProx unchanged but enables
    additional features such as: Saving global checkpoints, saving metrics to the local
    file system as a JSON."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Dictionary to store metrics per round
        self.results_to_save = {}

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        """Aggregate received model updates and metrics, then save global model checkpoint."""
        
        # Call the default aggregate_fit method from FedProx
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        # Convert parameters to ndarrays
        ndarrays = parameters_to_ndarrays(parameters_aggregated)
        
        # Instantiate model
        model = get_model()
        
        # Apply parameters to model
        model.set_weights(ndarrays)
        
        # Create directory if it doesn't exist
        os.makedirs("checkpoints", exist_ok=True)
        
        # Save the entire model
        model.save(f"checkpoints/global_model_round_{server_round}.keras")

        return parameters_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        """Evaluate global model, then save metrics to a local JSON file."""
        
        # Call the default evaluate method from FedProx
        loss, metrics = super().evaluate(server_round, parameters)
        
        # Store metrics as dictionary
        my_results = {"loss": loss, **metrics}
        self.results_to_save[server_round] = my_results
        
        # Save metrics as JSON
        with open("results.json", "w") as json_file:
            json.dump(self.results_to_save, json_file, indent=4)

        return loss, metrics
