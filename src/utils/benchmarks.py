import time

class BenchmarkSuite:
    def __init__(self):
        self.results = []
        
    def record(self, metric_name, value, unit=""):
        self.results.append({
            "metric": metric_name,
            "value": value,
            "unit": unit
        })
        
    def display_results(self):
        print("\n" + "="*40)
        print(" TeleZK-FL BENCHMARK RESULTS ")
        print("="*40)
        for res in self.results:
            val = f"{res['value']:.4f}" if isinstance(res['value'], float) else res['value']
            print(f"{res['metric']:<25}: {val} {res['unit']}")
        print("="*40 + "\n")

def measure_training_latency(client, epochs=1):
    start = time.time()
    client.train(epochs=epochs)
    return time.time() - start
