import subprocess
import os
import torch
from process_sensor_data.imuDataPipeline import full_sensor_pipeline
import streamlit as st # type: ignore

def run_model_inference_subprocess(input_path, output_path, checkpoint_path):
    """
    Run the inference using subprocess to call the command line directly.
    """
    # Set the path to your run_inference.py script
    inference_script = "application/run_inference.py"  # Adjust if in a different directory
    
    # Set the IMUPoser src path
    imuposer_src_path = "/content/IMUPoser/src"  # Change this to match your environment
    
    # Construct the command
    cmd = [
        "python3", 
        inference_script,
        "--checkpoint", checkpoint_path,
        "--input", input_path,
        "--output", output_path,
        "--device", "cpu"
    ]
    
    # Set the environment variable for PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{imuposer_src_path}:{env.get('PYTHONPATH', '')}"
    
    try:
        # Run the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=env
        )
        
        # Get output
        stdout, stderr = process.communicate()
        
        # Check results
        if process.returncode == 0:
            try:
                predictions = torch.load(output_path)
                return True, predictions
            except Exception as e:
                return True, "Inference successful but couldn't load predictions"
        else:
            return False, stderr
            
    except Exception as e:
        return False, str(e)
    


def process_uploaded_files(data_dir, output_dir='rawdata/processed', run_model=True, checkpoint_path=None):
    """ Process the uploaded IMU data files using the imuDataPipeline and run inference. """
    os.makedirs(output_dir, exist_ok=True)
    
    st.info("Starting pipeline processing...")
    
    # Run the full sensor pipeline (handles all detection and processing)
    st.text("Step 1: Processing sensor data...")
    synced_dfs, tensor = full_sensor_pipeline(data_dir=data_dir, output_path=os.path.join(output_dir, 'imuposer_data.pt'))
    
    # Determine which devices were active (non-None in the synced_dfs)
    device_names = ['phone', 'earbuds', 'left_watch', 'right_watch']
    active_device_indices = [i for i, df in enumerate(synced_dfs) if df is not None]
    active_devices = [device_names[i] for i in active_device_indices]
    
    # Convert device names to more readable format
    readable_device_names = []
    for device in active_devices:
        if device == "phone":
            readable_device_names.append("Phone")
        elif device == "earbuds":
            readable_device_names.append("Earbuds")
        elif device == "left_watch":
            readable_device_names.append("Left Watch")
        elif device == "right_watch":
            readable_device_names.append("Right Watch")
    
    st.text(f"✓ Successfully processed data from {len(readable_device_names)} device(s): {', '.join(readable_device_names)}")
    
    # Save tensor with metadata
    tensor_path = os.path.join(output_dir, 'imuposer_data.pt')
    torch.save({
        'imu_data': tensor,
        'active_devices': active_devices,
        'timestamp': torch.tensor([]),  # Add timestamp if available
    }, tensor_path)
    
    st.text(f"✓ Tensor shape: {tensor.shape}")
    st.text(f"✓ Saved to: {tensor_path}")
    
    predictions = None
    if run_model:
        st.text("Step 2: Running model inference...")
        
        # Use default checkpoint path if none provided
        if checkpoint_path is None:
            checkpoint_path = "model_checkpoints/imuposer_model.ckpt"
        
        # Set output path for predictions
        predictions_path = os.path.join(output_dir, 'predictions.pt')
        
        # Run inference using subprocess
        success, result = run_model_inference_subprocess(tensor_path, predictions_path, checkpoint_path)
        
        if success:
            if isinstance(result, torch.Tensor):
                predictions = result
                st.text(f"✓ Generated predictions with shape: {predictions.shape}")
            else:
                st.text(f"✓ Inference completed: {result}")
        else:
            st.error(f"✗ Model inference failed: {result}")
    
    return readable_device_names, tensor, predictions