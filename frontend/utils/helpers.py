import subprocess
import os
import torch
from process_sensor_data.imuDataPipeline import full_sensor_pipeline
import streamlit as st # type: ignore

def run_model_inference_subprocess(input_path, output_path, checkpoint_path, install_deps=True):
    """
    Run the inference using subprocess to call the command line directly.
    
    Args:
        input_path: Path to the input tensor file
        output_path: Path to save the output predictions
        checkpoint_path: Path to the model checkpoint
        install_deps: Whether to install dependencies if missing
        
    Returns:
        success: Boolean indicating if the process completed successfully
        message: Output message or error
    """
    
    st.text("Running model inference via command line...")
    
    # Make sure the directories exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Install missing dependencies if requested
    if install_deps:
        st.text("Checking and installing required dependencies...")
        try:
            # Try to install pytorch_lightning and any other necessary packages
            subprocess.check_call([
                "pip", "install", 
                "pytorch_lightning",
                "six",
                "einops",
                "torchvision",
                "--quiet"
            ])
            st.text("✓ Dependencies installed successfully")
        except Exception as e:
            st.warning(f"Could not install dependencies: {str(e)}")
    
    # Construct the command - similar to your original command
    # Adjust paths as necessary for your environment
    imuposer_src_path = "/content/IMUPoser"  # Update this path to match your environment
    
    # The command to run
    script_path = "/content/IMUPoser/application/run_inference.py"  # Update this to the correct path
    
    cmd = [
        "python3", 
        script_path,
        "--checkpoint", checkpoint_path,
        "--input", input_path,
        "--output", output_path,
        "--device", "cpu"
    ]
    
    # Set the environment variable for PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{imuposer_src_path}:{env.get('PYTHONPATH', '')}"
    
    st.text(f"Running command: {' '.join(cmd)}")
    st.text(f"Using PYTHONPATH: {env['PYTHONPATH']}")
    
    try:
        # Run the process with live output capture
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=env,
            bufsize=1  # Line buffered
        )
        
        # Capture and display output in real-time
        stdout_lines = []
        stderr_lines = []
        
        # Process stdout
        if process.stdout is not None:
            for line in process.stdout:
                stdout_lines.append(line.strip())
                st.text(f"[INFO] {line.strip()}")
        
        if process.stderr is not None:
            for line in process.stderr:
                stderr_lines.append(line.strip())
                st.text(f"[ERROR] {line.strip()}")
        
        # Wait for process to complete
        process.wait()
        
        # Check if the process was successful
        if process.returncode == 0:
            st.text("✓ Model inference completed successfully")
            
            # Try to load the predictions to return them
            try:
                predictions = torch.load(output_path)
                return True, predictions
            except Exception as e:
                return True, f"Inference successful but couldn't load predictions: {str(e)}"
        else:
            error_msg = "\n".join(stderr_lines)
            st.error(f"Command failed with return code {process.returncode}")
            return False, error_msg
            
    except Exception as e:
        st.error(f"Error running inference command: {str(e)}")
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
    
    tensor_path = os.path.join(output_dir, 'imuposer_data.pt')
    torch.save({
        'imu_data': tensor,
        'active_devices': active_devices,
        'timestamp': torch.tensor([]),
    }, tensor_path)
    
    st.text(f"✓ Tensor shape: {tensor.shape}")
    st.text(f"✓ Saved to: {tensor_path}")
    
    # Run model inference if requested
    predictions = None
    if run_model:
        st.text("Step 2: Running model inference...")
        
        # Use default checkpoint path if none provided
        if checkpoint_path is None:
            # Try to find the checkpoint in common locations
            possible_paths = [
                os.path.abspath("checkpoints/checkpoint.ckpt")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    checkpoint_path = path
                    st.text(f"Found checkpoint at: {checkpoint_path}")
                    break
            
            if checkpoint_path is None:
                st.warning("Could not find checkpoint file. Please specify the path.")
                return readable_device_names, tensor, None
        
        # Set output path for predictions
        predictions_path = os.path.join(output_dir, 'predictions.pt')
        
        # Run inference using subprocess with dependency installation
        success, result = run_model_inference_subprocess(
            tensor_path, 
            predictions_path, 
            checkpoint_path,
            install_deps=True  # Automatically install missing dependencies
        )
        
        if success:
            if isinstance(result, torch.Tensor):
                predictions = result
                st.text(f"✓ Generated predictions with shape: {predictions.shape}")
            else:
                st.text(f"✓ Inference completed: {result}")
            
            st.text(f"✓ Saved predictions to: {predictions_path}")
        else:
            st.error("✗ Model inference failed. Continuing with other processing steps.")
    
    return readable_device_names, tensor, predictions