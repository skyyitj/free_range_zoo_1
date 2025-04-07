import kagglehub

# Download latest version
path = kagglehub.dataset_download("picklecat/moasei-aamas-2025-competition-configurations")

print("Path to dataset files:", path)