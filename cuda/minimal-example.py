import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForImageClassification, AutoImageProcessor, pipeline
from pyJoules.device import DeviceFactory
from pyJoules.device.rapl_device import RaplPackageDomain, RaplDramDomain
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.energy_meter import EnergyMeter
from PIL import Image

def inference(pipe, image, num_requests):
    """Runs inference for a given number of requests and returns the output."""
    for _ in range(num_requests):
        output = pipe(image)
    return output

def measure_energy(model_name: str, num_requests: int = 10):
    """Measures energy consumption during inference for a ViT-based model over multiple requests."""
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    pipe = pipeline("image-classification", model=model, feature_extractor=processor, device=0 if torch.cuda.is_available() else -1)
    
    domains = [RaplPackageDomain(0), RaplDramDomain(0), NvidiaGPUDomain(0)]
    devices = DeviceFactory.create_devices(domains)
    meter = EnergyMeter(devices)
    
    image = Image.fromarray((torch.rand(224, 224, 3) * 255).byte().numpy())  # Convert random tensor to PIL image
    
    meter.start(tag='inference')
    inference(pipe, image, num_requests)
    meter.stop()
    
    trace = meter.get_trace()
    total_energy = {domain: sum(entry.energy[domain] for entry in trace if domain in entry.energy) for domain in trace[0].energy}
    avg_energy = {domain: total_energy[domain] / num_requests for domain in total_energy}
    
    print(f"Model: {model_name}, Avg Energy Consumption: {avg_energy} uJ")
    return model_name, avg_energy, model.num_parameters()

def visualize_results(results, selected_domain="package_0"):
    """Visualizes the energy consumption across multiple ViT models for a selected domain."""
    models, energies, params = zip(*results)
    
    plt.figure(figsize=(10, 6))
    domain_energies = [energy[selected_domain] for energy in energies]
    plt.bar(models, domain_energies, label=selected_domain)
    
    for i, txt in enumerate(params):
        plt.text(i, domain_energies[i] + 100, f"{txt/1e6:.1f}M", ha='center', fontsize=12)
    
    plt.xlabel("ViT Models")
    plt.ylabel("Avg Energy Consumption (uJ)")
    plt.title(f"Energy Consumption Across ViT Models ({selected_domain})")
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig(f"vit_energy_comparison_{selected_domain}.png")
    plt.show()

if __name__ == "__main__":
    models = ["google/vit-base-patch16-224", "google/vit-large-patch16-224", "google/vit-huge-patch14-224-in21k"]
    results = [measure_energy(model) for model in models]
    selected_domain = "nvidia_gpu_0"  # Change this to 'dram_0' or 'nvidia_gpu_0' as needed
    visualize_results(results, selected_domain)