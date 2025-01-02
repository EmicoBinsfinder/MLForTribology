import os
from itertools import product

# Define the hyperparameter ranges
num_elite_values = [5, 10, 15, 20, 25]  # Ensure these are less than generation_size_values
mutation_rate_values = [0.5, 0.7, 0.85, 0.95]
generation_size_values = [30, 50, 75, 100]
max_generations_values = [100, 250, 500, 1000, 2000]

# Base directory to save the generated directories and scripts
base_output_dir = "generated_parameter_combinations"
os.makedirs(base_output_dir, exist_ok=True)

# Template for the script
with open("GeneticAlgorithmML.py", "r") as template_file:
    template_script = template_file.read()

# Generate directories and scripts for each combination of hyperparameters
for num_elite, mutation_rate, generation_size, max_generations in product(
    num_elite_values, mutation_rate_values, generation_size_values, max_generations_values
):
    # Ensure NumElite is less than GenerationSize
    if num_elite >= generation_size:
        continue
    
    # Create a directory for this combination of hyperparameters
    dir_name = f"NumElite{num_elite}_MutRate{mutation_rate}_GenSize{generation_size}_MaxGen{max_generations}"
    output_dir = os.path.join(base_output_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # Modify the template with the current hyperparameter values
    modified_script = template_script
    modified_script = modified_script.replace("NumElite = 25", f"NumElite = {num_elite}")
    modified_script = modified_script.replace("MutationRate = 0.95", f"MutationRate = {mutation_rate}")
    modified_script = modified_script.replace("GenerationSize = 50", f"GenerationSize = {generation_size}")
    modified_script = modified_script.replace("MaxGenerations = 500", f"MaxGenerations = {max_generations}")

    # Save the modified script in the corresponding directory
    file_name = "GeneticAlgorithmML.py"
    with open(os.path.join(output_dir, file_name), "w") as output_file:
        output_file.write(modified_script)

print(f"Directories and scripts generated in {base_output_dir}")
