import torch
from pathlib import Path
import argparse
import sys

def load_model(checkpoint_path, device='cpu'):
    """
    Load the trained IMUPoser model from checkpoint
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to run inference on ('cpu' or 'cuda')
        
    Returns:
        Loaded model
    """
    # Make sure checkpoint exists
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Set device
    device = torch.device(device)
    
    # Add parent directory to system path
    root_dir = Path(__file__).resolve().parent.parent
    sys.path.append(str(root_dir))
    
    # Import necessary modules
    from imuposer.config import Config, amass_combos
    from src.imuposer.models.LSTMs.IMUPoser_Model import IMUPoserModel
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint or use default
    if 'hyper_parameters' in checkpoint and 'config' in checkpoint['hyper_parameters']:
        # For newer PyTorch Lightning checkpoints
        config_dict = checkpoint['hyper_parameters']['config']
        print("Using config from checkpoint's hyper_parameters")
    else:
        # Create a default config
        # You'll need to use the same config settings that were used for training
        # This is a guess based on your training script
        combo_id = 'global'  # Update this to match your trained model
        config = Config(
            experiment=f"IMUPoserGlobalModel_{combo_id}", 
            model="GlobalModelIMUPoser",
            project_root_dir=str(root_dir), 
            joints_set=amass_combos[combo_id], 
            normalize=False,
            r6d=True, 
            loss_type="mse", 
            use_joint_loss=True, 
            device=device
        )
        print("Using default config (might need adjustment)")
    
    # Create model instance
    if 'hyper_parameters' in checkpoint and 'config' in checkpoint['hyper_parameters']:
        model = IMUPoserModel(config_dict)
    else:
        model = IMUPoserModel(config)
    
    # Load the weights
    if 'state_dict' in checkpoint:
        # PyTorch Lightning saves the model state in 'state_dict'
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        # Some models save the state in 'model_state_dict'
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("WARNING: Could not find state_dict in checkpoint")
    
    # Set to evaluation mode
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    return model

def run_inference(model, input_tensor_path, output_path=None, device='cpu'):
    """
    Run inference with the loaded model
    
    Args:
        model: Loaded IMUPoser model
        input_tensor_path: Path to the input tensor (.pt file)
        output_path: Path to save the output
        device: Device to run inference on ('cpu' or 'cuda')
        
    Returns:
        Model predictions (SMPL poses)
    """
    # Set device
    device = torch.device(device)
    
    # Load input tensor
    input_tensor_path = Path(input_tensor_path)
    if not input_tensor_path.exists():
        raise FileNotFoundError(f"Input tensor not found at {input_tensor_path}")
    
    print(f"Loading input tensor from: {input_tensor_path}")
    input_data = torch.load(input_tensor_path)
    
    # Extract the IMU data tensor
    if isinstance(input_data, dict) and 'imu_data' in input_data:
        input_tensor = input_data['imu_data']
    else:
        input_tensor = input_data
    
    if isinstance(input_tensor, torch.Tensor):
        input_tensor = input_tensor.to(device)
    else:
        raise TypeError("Input data is not a tensor")
    
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Prepare input for model
    # Add batch dimension if needed
    if len(input_tensor.shape) == 2:
        input_tensor = input_tensor.unsqueeze(0)
    
    # Get sequence length for each batch (all frames in this case)
    seq_length = input_tensor.shape[1]
    input_lengths = torch.tensor([seq_length], device=device)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        predictions = model(input_tensor, input_lengths)
    
    # Process predictions (depends on model output format)
    # Based on the model code, we're selecting the pose parameters
    if isinstance(predictions, tuple) and len(predictions) >= 1:
        pose_predictions = predictions[0]
    else:
        pose_predictions = predictions
    
    # Select pose parameters if needed (based on your model architecture)
    if hasattr(model, 'n_pose_output') and isinstance(pose_predictions, torch.Tensor) and pose_predictions.shape[-1] > model.n_pose_output:
        pose_predictions = torch.tensor(pose_predictions)[:, :, :model.n_pose_output]
    
    # Process and save output
    if output_path:
        output_path = Path(output_path)
        print(f"Saving predictions to: {output_path}")
        torch.save(pose_predictions, output_path)
    
    return pose_predictions

def main():
    parser = argparse.ArgumentParser(description="Run inference with IMUPoser model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, default="mobileposer_data.pt", help="Path to input tensor")
    parser.add_argument("--output", type=str, default="pose_predictions.pt", help="Path to save output")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu or cuda)")
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    
    # Run inference
    predictions = run_inference(model, args.input, args.output, args.device)
    
    # Print predictions summary
    if isinstance(predictions, tuple):
        print(f"Generated predictions with shape: {[p.shape for p in predictions]}")
    else:
        print(f"Generated predictions with shape: {predictions.shape}")
    
    print("Inference completed successfully!")

if __name__ == "__main__":
    main()