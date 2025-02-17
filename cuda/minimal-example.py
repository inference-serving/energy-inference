import torch
import matplotlib.pyplot as plt
import pynvml
import time
from transformers import AutoModelForImageClassification, AutoImageProcessor, pipeline
from pyJoules.device import DeviceFactory
from pyJoules.device.rapl_device import RaplPackageDomain, RaplDramDomain
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.energy_meter import EnergyMeter
from PIL import Image

def inference(pipe, image, num_requests):
    """Runs inference for a given number of requests and returns the output."""
    latencies = []
    for _ in range(num_requests):
        start_time = time.time()
        output = pipe(image)
        end_time = time.time()
        latencies.append(end_time - start_time)
    avg_latency = sum(latencies) / num_requests
    return output, avg_latency

def measure_memory_usage():
    """Measures GPU memory usage using pynvml."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / (1024 ** 2)  # Convert to MB

def measure_energy(model_name: str, num_requests: int = 60, selected_energy_domain="nvidia_gpu_0"):
    """Measures energy consumption, latency, and memory usage during inference."""
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    pipe = pipeline("image-classification", model=model, feature_extractor=processor, device=0 if torch.cuda.is_available() else -1)
    
    domains = [RaplPackageDomain(0), RaplDramDomain(0), NvidiaGPUDomain(0)]
    devices = DeviceFactory.create_devices(domains)
    meter = EnergyMeter(devices)
    
    image = Image.fromarray((torch.rand(224, 224, 3) * 255).byte().numpy())  # Convert random tensor to PIL image
    
    meter.start(tag='inference')
    output, avg_latency = inference(pipe, image, num_requests)
    meter.stop()
    
    trace = meter.get_trace()
    total_energy = {domain: sum(entry.energy[domain] for entry in trace if domain in entry.energy) for domain in trace[0].energy}
    avg_energy = {domain: total_energy[domain] / num_requests for domain in total_energy}
    memory_usage = measure_memory_usage()
    
    # Updated accuracy values from reliable sources
    accuracy = {"vit-base": 81.1, "vit-large": 84.8, "vit-huge": 87.0}  # Updated values
    short_model_name = model_name.split('/')[-1].replace("vit-base-patch16-224", "vit-base").replace("vit-large-patch16-224", "vit-large").replace("vit-huge-patch14-224-in21k", "vit-huge")
    
    selected_energy = avg_energy.get(selected_energy_domain, 0)
    print(f"Model: {short_model_name}, {selected_energy_domain} Energy Consumption: {selected_energy} uJ, Avg Latency: {avg_latency}s, Memory Usage: {memory_usage} MB, Accuracy: {accuracy.get(short_model_name, 'N/A')}%")
    return short_model_name, selected_energy, avg_latency, memory_usage, accuracy.get(short_model_name, 'N/A'), model.num_parameters()

def visualize_results(results, selected_metric):
    """Visualizes the selected metric across multiple ViT models."""
    models, energies, latencies, memory_usages, accuracies, params = zip(*results)
    metrics = {"energy": energies, "latency": latencies, "memory": memory_usages, "accuracy": accuracies}
    
    plt.figure(figsize=(10, 6))
    plt.bar(models, metrics[selected_metric], label=selected_metric)
    
    for i, txt in enumerate(params):
        plt.text(i, metrics[selected_metric][i] + 100, f"{txt/1e6:.1f}M", ha='center', fontsize=12)
    
    plt.xlabel("ViT Models")
    plt.ylabel(f"{selected_metric.capitalize()} Measurement")
    plt.title(f"{selected_metric.capitalize()} Across ViT Models")
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig(f"vit_{selected_metric}_comparison.png")
    plt.show()

if __name__ == "__main__":
    models = ["google/vit-base-patch16-224", "google/vit-large-patch16-224", "google/vit-huge-patch14-224-in21k"]
    selected_energy_domain = "nvidia_gpu_0"  # Change to 'package_0' or 'dram_0' as needed
    results = [measure_energy(model, selected_energy_domain=selected_energy_domain) for model in models]
    
    for metric in ["energy", "latency", "memory", "accuracy"]:
        visualize_results(results, metric)
