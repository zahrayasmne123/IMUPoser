import torch
from pathlib import Path

from src.imuposer.config import Config, amass_combos
from src.imuposer.models.LSTMs.IMUPoser_Model import IMUPoserModel

base_dir = "/dcs/22/u2254377/cs310/IMUPoser"

def load_model(checkpoint_path, device='cpu'):
    # Disable CUDA explicitly
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    try:
        # Make sure checkpoint exists
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        print(f"Loading model from checkpoint: {checkpoint_path}")
        
        # Force CPU device
        device = torch.device('cpu')
        
        # Load checkpoint with CPU mapping
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        # Determine configuration
        if 'hyper_parameters' in checkpoint and 'config' in checkpoint['hyper_parameters']:
            config_dict = checkpoint['hyper_parameters']['config']
            
            # Explicitly create Config with CPU device
            config = Config(
                experiment=config_dict.experiment if hasattr(config_dict, 'experiment') else "IMUPoserGlobalModel",
                model=config_dict.model if hasattr(config_dict, 'model') else "GlobalModelIMUPoser",
                project_root_dir= base_dir,
                joints_set=config_dict.joints_set if hasattr(config_dict, 'joints_set') else amass_combos['global'],
                normalize=False,
                r6d=True,
                loss_type="mse",
                use_joint_loss=True,
                device='cpu',
                og_smpl_model_path=  os.path.join(base_dir, "src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")
            )
        else:
            # Fallback configuration
            config = Config(
                experiment="IMUPoserGlobalModel",
                model="GlobalModelIMUPoser",
                project_root_dir=base_dir,
                joints_set=amass_combos['global'],
                normalize=False,
                r6d=True,
                loss_type="mse",
                use_joint_loss=True,
                device='cpu',
                og_smpl_model_path=  os.path.join(base_dir, "src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")
            )
        
        # Verify SMPL model path
        if not config.og_smpl_model_path.exists():
            raise FileNotFoundError(f"SMPL model not found at {config.og_smpl_model_path}")
        
        # Create model instance
        model = IMUPoserModel(config)
        
        # Load weights
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("WARNING: Could not find state_dict in checkpoint")
        
        # Ensure model is in evaluation mode and on CPU
        model = model.to(torch.device('cpu'))
        model.eval()
        
        print("Model loaded successfully")
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
def generate_prediction(model, input_tensor_path, output_path=None, device='cpu'):
    """
    Run inference with the loaded model
    """
    # Force device to CPU
    device = torch.device('cpu')
    
    # Load input tensor
    input_tensor_path = Path(input_tensor_path)
    if not input_tensor_path.exists():
        raise FileNotFoundError(f"Input tensor not found at {input_tensor_path}")
    
    print(f"Loading input tensor from: {input_tensor_path}")
    
    # Load tensor with CPU mapping
    input_data = torch.load(input_tensor_path, map_location=torch.device('cpu'))
    
    # Extract the IMU data tensor
    if isinstance(input_data, dict) and 'imu_data' in input_data:
        input_tensor = input_data['imu_data']
    else:
        input_tensor = input_data
    
    # Ensure tensor is on CPU and float type
    if isinstance(input_tensor, torch.Tensor):
        input_tensor = input_tensor.to(device).float()
    else:
        raise TypeError("Input data is not a tensor")
    
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Prepare input for model
    # Add batch dimension if needed
    if len(input_tensor.shape) == 2:
        input_tensor = input_tensor.unsqueeze(0)
    
    # Get sequence length for each batch (all frames in this case)
    seq_length = input_tensor.shape[1]
    input_lengths = torch.tensor([seq_length], device=device, dtype=torch.long)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        # Ensure model is in evaluation mode and on CPU
        model.eval()
        model = model.to(device)
        
        predictions = model(input_tensor, input_lengths)
    
    # Process predictions (depends on model output format)
    # Based on the model code, we're selecting the pose parameters
    if isinstance(predictions, tuple) and len(predictions) >= 1:
        pose_predictions = predictions[0]
    else:
        pose_predictions = predictions
    
    # Ensure predictions are on CPU and float type
    pose_predictions = pose_predictions.to(device).float() # type: ignore
    
    # Select pose parameters if needed (based on your model architecture)
    if hasattr(model, 'n_pose_output') and isinstance(pose_predictions, torch.Tensor) and pose_predictions.shape[-1] > model.n_pose_output:
        pose_predictions = pose_predictions[:, :, :model.n_pose_output]
    
    # Process and save output
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        print(f"Saving predictions to: {output_path}")
        try:
            torch.save(pose_predictions, output_path)
            print(f"Successfully saved predictions with shape {pose_predictions.shape}")
        except Exception as e:
            print(f"Error saving predictions: {str(e)}")
    
    return pose_predictions