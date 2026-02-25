import torch
import numpy as np
from src.data.health_data import HealthDataGenerator, get_dataloader
from src.fl.client import FLClient, SimpleModel
from src.fl.server import FLServer
from src.zkp.lut_zkp import LUTZKPSimulator
from src.utils.benchmarks import BenchmarkSuite
import warnings

warnings.filterwarnings("ignore")

def run_simulation():
    # init sim
    print("Setting up TeleZK-FL Simulation...")
    num_clients = 3
    num_rounds = 5
    num_features = 5
    
    data_gen = HealthDataGenerator(num_samples=200, num_features=num_features)
    global_model = SimpleModel(input_dim=num_features)
    server = FLServer(global_model)
    zkp_sim = LUTZKPSimulator()
    benchmarks = BenchmarkSuite()
    
    # federated learning loop
    for r in range(num_rounds):
        print(f"\n--- Round {r+1}/{num_rounds} ---")
        client_updates = []
        client_scales = []
        client_proofs = []
        
        for c in range(num_clients):
            # gen local data
            # FIXME: memory leak here for large num_samples?
            X, y = data_gen.generate_data()
            loader = get_dataloader(X, y, batch_size=32)
            
            # Create client with current global weights
            client = FLClient(c, loader, model=SimpleModel(input_dim=num_features))
            client.model.load_state_dict(server.get_model().state_dict())
            
            # local training
            _ = client.train(epochs=1)
            
            # get quant updates & proof
            updates, scales = client.get_updates()
            proof = client.generate_proof()
            
            client_updates.append(updates)
            client_scales.append(scales)
            client_proofs.append(proof)
            print(f"  Client {c} trained and quantized INT8 updates.")
            
        # Server Aggregation
        if server.verify_proofs(client_proofs):
            server.aggregate(client_updates, client_scales)
            print("  Server verified proofs and aggregated updates.")
        else:
            print("  Server failed to verify ZK proofs!")

    # run benchmarks
    print("\nRunning ZK Performance Benchmarks...")
    # Calculate model size (number of parameters)
    model_params = sum(p.numel() for p in global_model.parameters())
    zk_results = zkp_sim.get_benchmarks(model_params)
    
    # save results
    benchmarks.record("Model Parameters", model_params)
    benchmarks.record("Quantization", "INT8")
    benchmarks.record("ZK Speedup (Target)", zk_results["speedup"], "x")
    benchmarks.record("Energy Reduction (Target)", zk_results["energy_reduction"], "x")
    benchmarks.record("Standard Proof Latency", zk_results["standard_time"], "s")
    benchmarks.record("TeleZK-FL Proof Latency", zk_results["lut_time"], "s")
    benchmarks.record("Standard Proof Energy", zk_results["standard_energy"], "J")
    benchmarks.record("TeleZK-FL Proof Energy", zk_results["lut_energy"], "J")
    
    # Print Output
    benchmarks.display_results()
    
    print("\nSimulation Complete. Results mirror the performance claims of TeleZK-FL.")

if __name__ == "__main__":
    run_simulation()
