import os
import subprocess



# Directory containing the generated training scripts
scripts_dir = 'generated_scripts'
submission_dir = 'submission_scripts'

# Create a directory to store the submission scripts
os.makedirs(submission_dir, exist_ok=True)

# Define the base submission script template
submission_template = """#!/bin/bash -l

module -f unload compilers mpi gcc-libs
module load beta-modules
module load gcc-libs/10.2.0
module load python3/3.9-gnu-10.2.0
module load cuda/11.3.1/gnu-10.2.0
module load cudnn/8.2.1.32/cuda-11.3
module load pytorch/1.11.0/gpu

pip3 install torch-scatter

module load python3/recommended
source activate gt4sd

# Request ten minutes of wallclock time (formathours:minutes:seconds).
#$ -l h_rt=47:59:59

# Submit a job to the GPU nodes by adding a request for a number of GPUs per node
#$ -l gpu=2

# Only Free jobs are available at present. Use your normal projects
#$ -P Free
#$ -A Imperial_Meng
# Set the name of the job.
#$ -N MPNNTraining_{script_name}

#$ -pe mpi 2

# Set the working directory to somewhere in your scratch space.
#$ -wd /lustre/scratch/mmm1058/MLForTribology/PropertyPrediction/GraphPrediction/MPNN

# Run our MPI job.  GERun is a wrapper that launchesMPI jobs on our clusters.

python3 {script_path}
"""

# List all the training scripts in the directory
training_scripts = [f for f in os.listdir(scripts_dir) if f.endswith('.py')]

# Create and submit submission scripts for each training script
for script in training_scripts:
    script_name = os.path.splitext(script)[0]
    script_path = os.path.join(os.getcwd(), scripts_dir, script)
    print(script_path)
    submission_content = submission_template.format(script_name=script_name, script_path=script_path)
    submission_filename = os.path.join(os.getcwd(), submission_dir, f'submit_{script_name}.sh')
    with open(submission_filename, 'w') as submission_file:
        submission_file.write(submission_content)
    
    # Submit the job script
    result = subprocess.run(['qsub', submission_filename], capture_output=True, text=True)
    print(f'Submitted {script_name}')
    if result.returncode != 0:
        print(f"Error submitting {submission_filename}: {result.stderr}")
    else:
        print(f"Successfully submitted {submission_filename}: {result.stdout}")
