import subprocess
import os
import torch # type: ignore
from preprocessing.imuDataPipeline import full_sensor_pipeline
import streamlit as st # type: ignore
from post_processing.generate_predictions import load_model, generate_prediction
from post_processing.visualisepose import full_visualisation_pipeline

################ HELPER FUNCTIONS ################
#  Helper functions used accorss the application, helping to handle setting up the environment, 
# processing sensor data files, running model generation, and generating visualisations.
# 1. Run Setup Script: Minimal IMUPoser setup, looking for src directory and installs the IMUPoser package 
#    without reinstalling dependencies
# 2. Process Uploaded Files: Main function running the preprocessing pipeline stages to prepare data 
#    before putting it into IMUPoser deep learning model. Creates output directories if they don't exist
#    Calls full_visualisation_pipeline() to prepare the 3D pose visualisation
#

def run_setup_script():

    try:
        # Find the IMUPoser src directory
        potential_source_directories = ["./src", "../src", "/IMUPoser/src", "/src"]
        source_directory = None
        
        for directory in potential_source_directories:
            if os.path.exists(directory) and os.path.isdir(directory):
                source_directory = directory
                break
        
        if not source_directory:
            st.error("Could not find IMUPoser src directory")
            return False
        
        # Install the IMUPoser package without reinstalling dependencies
        subprocess.check_call([
            "pip", "install", "-e", source_directory,
            "--no-dependencies",  # Skip reinstalling dependencies
            "--quiet"
        ])
        
        st.success("✓ IMUPoser environment ready")
        return True
            
    except Exception:
        st.error("Error during minimal setup")
        return False

def process_uploaded_files(data_dir, output_dir='rawdata/processed', run_model=True, checkpoint_path=None):
    os.makedirs(output_dir, exist_ok=True)
    
    st.info("Starting pipeline processing...")
    
    # Run the full sensor pipeline (from preprocessing directory)
    st.text("Step 1: Processing sensor data...")
    synced_dfs, tensor = full_sensor_pipeline(data_dir=data_dir, output_path=os.path.join(output_dir, 'imuposer_data.pt'))
    
    # determine active devices from uploaded data files 
    device_names = ['phone', 'earbuds', 'left_watch', 'right_watch']
    active_device_indices = [i for i, df in enumerate(synced_dfs) if df is not None]
    active_devices = [device_names[i] for i in active_device_indices]
    
    # convert device names for readability 
    readable_device_names = []
    for individual_device in active_devices:
        if individual_device == "phone":
            readable_device_names.append("Phone")
        elif individual_device == "earbuds":
            readable_device_names.append("Earbuds")
        elif individual_device == "left_watch":
            readable_device_names.append("Left Watch")
        elif individual_device == "right_watch":
            readable_device_names.append("Right Watch")
    
    st.text(f"Successfully processed data from {len(readable_device_names)} device(s): {', '.join(readable_device_names)}")
    
    # Save tensor
    tensor_path = os.path.join(output_dir, 'imuposer_data.pt')
    torch.save({
        'imu_data': tensor,
        'active_devices': active_devices,
        'timestamp': torch.tensor([]),
    }, tensor_path)
    
    st.text(f"Tensor shape: {tensor.shape}")
    st.text(f"Saved to: {tensor_path}")
    
    predictions = None
    if run_model:
        st.text("Step 2: Making predictions with Model...")
        model_setup_sucessful = run_setup_script()
        
        if not model_setup_sucessful:
            st.warning("Environment setup had issues. Attempting to continue anyway...")
        
        if checkpoint_path is None:
            possible_paths = [
                os.path.abspath("checkpoints/checkpoint.ckpt")]
        
            for path_option in possible_paths:
                if os.path.exists(path_option):
                    checkpoint_path = path_option
                    st.text(f"Found checkpoint at: {checkpoint_path}")
                    break
            else:
                st.warning("Could not find checkpoint file automatically.")
                st.info("Please upload or specify the correct checkpoint path.")
                
               
        #  output directory path for predictions
        predictions_path = os.path.join(output_dir, 'predictions.pt')


        try:
            model = load_model(checkpoint_path)
            predictions = generate_prediction(model, tensor_path, predictions_path)
            
            st.text(f" Generated predictions with shape: {predictions.shape}") # type: ignore
                
        except ImportError:
                st.text("Import error: trying subprocess method")
                
                # Use the subprocess method as fallback
                path_to_script = None
                for path_option in ["./post_processing/generate_prediction.py", "../post_processing/generate_prediction.py", "./generate_prediction.py"]:
                    if os.path.exists(path_option):
                        path_to_script = path_option
                        break
                
                if path_to_script:
                    st.text(f"Found inference script at: {path_to_script}")
                    
                    # Create command
                    cmd = [
                        "python", path_to_script,
                        "--checkpoint", checkpoint_path,
                        "--input", tensor_path,
                        "--output", predictions_path,
                        "--device", "cpu"
                    ]
                    
                    st.text(f"Running command: {' '.join(cmd)}")
                    
                    # Run the command
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True
                    )
                    
                    # Process output
                    for line in iter(process.stdout.readline, ''): # type: ignore
                        if not line:
                            break
                        st.text(f"[INFO] {line.strip()}")
                    
                    for line in iter(process.stderr.readline, ''): # type: ignore
                        if not line:
                            break
                        st.text(f"[ERROR] {line.strip()}")
                    
                    # Wait for process to complete
                    process.wait()
                    
                    if process.returncode == 0:
                        st.text("Model inference completed successfully")
                        try:
                            predictions = torch.load(predictions_path)
                            st.text(f"Loaded predictions with shape: {predictions.shape}")
                        except Exception:
                            st.error("Error loading predictions")
                    else:
                        st.error("Command failed with return code")
                else:
                    st.error("Could not find inference script")
            
        except Exception:
            st.error("Error model predictions")

    full_visualisation_pipeline()
    
    return readable_device_names, tensor, predictions


