import subprocess
import os
import torch
from process_sensor_data.imuDataPipeline import full_sensor_pipeline
import streamlit as st # type: ignore
import tempfile

def run_setup_script():
    """
    Run the setup commands for IMUPoser as a shell script.
    This function mimics the conda environment setup process
    without actually creating a conda environment.
    """
    st.text("Setting up IMUPoser environment...")
    
    # Create a temporary shell script
    with tempfile.NamedTemporaryFile(suffix='.sh', delete=False) as temp:
        temp_path = temp.name
        
        # Write setup commands to the temp file
        script_content = """#!/bin/bash
# Install PyTorch and dependencies
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1

# Find the IMUPoser src directory
SRC_DIR=""
for dir in "./src" "../src" "/content/IMUPoser/src" "/content/src"; do
    if [ -d "$dir" ]; then
        SRC_DIR="$dir"
        break
    fi
done

if [ -z "$SRC_DIR" ]; then
    echo "ERROR: Could not find IMUPoser src directory"
    exit 1
fi

echo "Found IMUPoser src directory at: $SRC_DIR"

# Install requirements if requirements.txt exists
REQ_FILE=""
for file in "./requirements.txt" "../requirements.txt" "/content/IMUPoser/requirements.txt" "/content/requirements.txt"; do
    if [ -f "$file" ]; then
        REQ_FILE="$file"
        break
    fi
done

if [ -n "$REQ_FILE" ]; then
    echo "Installing requirements from: $REQ_FILE"
    pip install -r "$REQ_FILE"
fi

# Install the IMUPoser package from source
pip install -e "$SRC_DIR"

echo "IMUPoser environment setup complete!"
"""
        temp.write(script_content.encode())
    
    # Make the script executable
    os.chmod(temp_path, 0o755)
    
    try:
        # Run the script
        process = subprocess.Popen(
            ['/bin/bash', temp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Process stdout in real-time
        if process.stdout is not None:
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break
                st.text(f"[SETUP] {line.strip()}")
        
        # Process stderr in real-time
        if process.stderr is not None:
            for line in iter(process.stderr.readline, ''):
                if not line:
                    break
                st.text(f"[ERROR] {line.strip()}")
        
        # Wait for the process to complete
        process.wait()
        
        # Check if successful
        if process.returncode == 0:
            st.success("✓ IMUPoser environment setup completed successfully")
            return True
        else:
            st.error(f"✗ IMUPoser environment setup failed with return code {process.returncode}")
            return False
            
    except Exception as e:
        st.error(f"Error during environment setup: {str(e)}")
        return False
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def run_model_inference_subprocess(input_path, output_path, checkpoint_path, install_deps=True):
    """
    Run the inference using subprocess to call the command line directly.
    """
    st.text("Running model inference via command line...")
    
    # Make sure the directories exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Install missing dependencies if requested
    if install_deps:
        st.text("Installing required dependencies...")
        try:
            # Install pytorch_lightning and other dependencies
            subprocess.check_call([
                "pip", "install", 
                "pytorch_lightning==1.5.0",
                "einops",
                "torchvision",
                "--quiet"
            ])
            st.text("✓ Basic dependencies installed")
            
            # Now, install the src directory as a package
            src_path = "/content/IMUPoser/src"  # Update this to match your environment
            
            if os.path.exists(src_path):
                st.text(f"Installing IMUPoser package from {src_path}...")
                subprocess.check_call([
                    "pip", "install", "-e", src_path,
                    "--quiet"
                ])
                st.text("✓ IMUPoser package installed successfully")
            else:
                st.warning(f"Source directory not found at {src_path}")
                
                # Try to find src directory in common locations
                possible_src_paths = [
                    "./src",
                    "../src",
                    "/content/src",
                    "./IMUPoser/src"
                ]
                
                for path in possible_src_paths:
                    if os.path.exists(path) and os.path.isdir(path):
                        st.text(f"Found source directory at {path}, installing...")
                        subprocess.check_call([
                            "pip", "install", "-e", path,
                            "--quiet"
                        ])
                        st.text(f"✓ IMUPoser package installed from {path}")
                        break
                else:
                    st.error("Could not find IMUPoser source directory to install")
                    
        except Exception as e:
            st.warning(f"Could not install dependencies: {str(e)}")
    
    # The command to run
    script_path = "/content/IMUPoser/application/run_inference.py"  # Update this to the correct path
    
    # Check if the script exists at the specified path
    if not os.path.exists(script_path):
        # Try to find the script in common locations
        possible_script_paths = [
            "./run_inference.py",
            "./application/run_inference.py",
            "../application/run_inference.py",
            "/content/run_inference.py"
        ]
        
        for path in possible_script_paths:
            if os.path.exists(path):
                script_path = path
                st.text(f"Found inference script at: {script_path}")
                break
        else:
            st.error("Could not find run_inference.py script. Please specify the correct path.")
            return False, "Inference script not found"
    
    # The command to run
    cmd = [
        "python", 
        script_path,
        "--checkpoint", checkpoint_path,
        "--input", input_path,
        "--output", output_path,
        "--device", "cpu"
    ]
    
    # Show command
    st.text(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
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
        
        # Check if the process was successful
        if process.returncode == 0:
            st.text("✓ Model inference completed successfully")
            
            # Try to load the predictions
            try:
                predictions = torch.load(output_path)
                return True, predictions
            except Exception as e:
                return True, f"Inference successful but couldn't load predictions: {str(e)}"
        else:
            st.error(f"Command failed with return code {process.returncode}")
            return False, "Inference failed"
            
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
    
    # Save tensor
    tensor_path = os.path.join(output_dir, 'imuposer_data.pt')
    torch.save({
        'imu_data': tensor,
        'active_devices': active_devices,
        'timestamp': torch.tensor([]),
    }, tensor_path)
    
    st.text(f"✓ Tensor shape: {tensor.shape}")
    st.text(f"✓ Saved to: {tensor_path}")
    
    predictions = None
    if run_model:
        st.text("Step 2: Running model inference...")
        
        # Set up environment before running inference
        # This will install the IMUPoser package and dependencies
        success = run_setup_script()
        
        if not success:
            st.warning("Environment setup had issues. Attempting to continue anyway...")
        
        # Use default checkpoint path if none provided
        if checkpoint_path is None:
            # Try to find the checkpoint in common locations
            possible_paths = [
                os.path.abspath("checkpoint.ckpt"),  # Current directory
                os.path.abspath("checkpoints/checkpoint.ckpt"),  # Checkpoints folder
                os.path.abspath("./model_checkpoints/imuposer_model.ckpt"),
                os.path.abspath("/content/checkpoint.ckpt"),  # Colab root
                os.path.abspath("/content/IMUPoser/checkpoint.ckpt"),  # Colab project dir
                os.path.abspath("/content/IMUPoser/checkpoints/checkpoint.ckpt")  # Nested dir
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    checkpoint_path = path
                    st.text(f"Found checkpoint at: {checkpoint_path}")
                    break
            else:
                st.warning("Could not find checkpoint file automatically.")
                st.info("Please upload or specify the correct checkpoint path.")
                
                # Allow user to upload a checkpoint file
                uploaded_checkpoint = st.file_uploader(
                    "Upload checkpoint file (.ckpt)", 
                    type=["ckpt", "pt"], 
                    key="checkpoint_uploader"
                )
                
                if uploaded_checkpoint:
                    # Save the uploaded checkpoint to a temporary location
                    temp_checkpoint_path = os.path.join(output_dir, "uploaded_checkpoint.ckpt")
                    with open(temp_checkpoint_path, "wb") as f:
                        f.write(uploaded_checkpoint.getbuffer())
                    
                    st.text(f"Using uploaded checkpoint: {temp_checkpoint_path}")
                    checkpoint_path = temp_checkpoint_path
                else:
                    st.error("No checkpoint available. Skipping inference.")
                    return readable_device_names, tensor, None
        
        # Set output path for predictions
        predictions_path = os.path.join(output_dir, 'predictions.pt')
        
        # Run inference using our updated subprocess function
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
            
            # Set session state flag for successful prediction
            st.session_state.predictions_available = True
        else:
            st.error("✗ Model inference failed. Continuing with other processing steps.")
            st.text("Please check the error message above for more details.")
    
    return readable_device_names, tensor, predictions