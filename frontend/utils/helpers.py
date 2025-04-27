import subprocess
import os
import torch # type: ignore
from preprocessing.imuDataPipeline import full_sensor_pipeline
import streamlit as st # type: ignore
from post_processing.run_inference import load_model, run_inference
from post_processing.visualisepose import visuals_pipeline

def run_setup_script():
    """
    Run the minimal setup commands for IMUPoser.
    This simplified version assumes most dependencies are already installed.
    """
    # st.text("Setting up IMUPoser environment...")
    
    try:
        # Find the IMUPoser src directory
        src_dirs = ["./src", "../src", "/IMUPoser/src", "/src"]
        src_dir = None
        
        for dir in src_dirs:
            if os.path.exists(dir) and os.path.isdir(dir):
                src_dir = dir
                # st.text(f"Found IMUPoser src directory at: {src_dir}")
                break
        
        if not src_dir:
            st.error("Could not find IMUPoser src directory")
            return False
        
        # Install the IMUPoser package without reinstalling dependencies
        subprocess.check_call([
            "pip", "install", "-e", src_dir,
            "--no-dependencies",  # Skip reinstalling dependencies
            "--quiet"
        ])
        
        st.success("✓ IMUPoser environment ready")
        return True
            
    except Exception as e:
        st.error(f"Error during minimal setup: {str(e)}")
        return False

def run_model_inference_subprocess(input_path, output_path, checkpoint_path, install_deps=True):
    """
    Run the inference using subprocess to call the command line directly.
    """
    import subprocess
    import os
    
    st.text("Running model inference via command line...")
    
    # Make sure the directories exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if install_deps:
        st.text("Verifying IMUPoser installation...")
        try:
            # Find and install the src directory as a package if needed
            src_paths = ["./src", "../src", "/IMUPoser/src", "./IMUPoser/src"]
            
            for path in src_paths:
                if os.path.exists(path) and os.path.isdir(path):
                    st.text(f"Found source directory at {path}")
                    # Only install if not already installed
                    try:
                        st.text("✓ IMUPoser package already installed")
                        break
                    except ImportError:
                        subprocess.check_call([
                            "pip", "install", "-e", path,
                            "--no-dependencies",
                            "--quiet"
                        ])
                        st.text(f"✓ IMUPoser package installed from {path}")
                        break
            else:
                st.warning("Could not find IMUPoser source directory")
                    
        except Exception as e:
            st.warning(f"Setup notice: {str(e)}")


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
            possible_paths = [
                os.path.abspath("checkpoints/checkpoint.ckpt")]
        
            for path in possible_paths:
                if os.path.exists(path):
                    checkpoint_path = path
                    st.text(f"Found checkpoint at: {checkpoint_path}")
                    break
            else:
                st.warning("Could not find checkpoint file automatically.")
                st.info("Please upload or specify the correct checkpoint path.")
                
               
        # Set output path for predictions
        predictions_path = os.path.join(output_dir, 'predictions.pt')


        try:
            # st.text("Imported inference functions directly")
            
            model = load_model(checkpoint_path, device='cpu')
            predictions = run_inference(model, tensor_path, predictions_path, device='cpu')
            
            st.text(f"✓ Generated predictions with shape: {predictions.shape}") # type: ignore
            # st.text(f"✓ Saved predictions to: {predictions_path}")
                
        except ImportError as e:
                st.text(f"Import error: {e}, trying subprocess method")
                
                # Use the subprocess method as fallback
                script_path = None
                for path in ["./post_processing/run_inference.py", "../post_processing/run_inference.py", "./run_inference.py"]:
                    if os.path.exists(path):
                        script_path = path
                        break
                
                if script_path:
                    st.text(f"Found inference script at: {script_path}")
                    
                    # Create command
                    cmd = [
                        "python", script_path,
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
                        st.text("✓ Model inference completed successfully")
                        try:
                            predictions = torch.load(predictions_path)
                            st.text(f"✓ Loaded predictions with shape: {predictions.shape}")
                        except Exception as e:
                            st.error(f"Error loading predictions: {e}")
                    else:
                        st.error(f"Command failed with return code {process.returncode}")
                else:
                    st.error("Could not find inference script")
            
        except Exception as e:
            st.error(f"Error during inference: {str(e)}")
            import traceback
            st.text(traceback.format_exc())


    visuals_pipeline()
    
    return readable_device_names, tensor, predictions


