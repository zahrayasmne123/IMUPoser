import torch
from pathlib import Path
from src.imuposer.config import Config, amass_combos, BASE_DIR
from src.imuposer.models.LSTMs.IMUPoser_Model import IMUPoserModel
import os

""" GENERATE PREDICTIONS.PY: Main file handling loading a pre-trained LSTM and runs inference to produce
pose predictions using the model as a 'black box'

1. Load Model: This function loads a pre-trained IMUPoser from a saved checkpoint file. Function 
configures the model to run on CPU and loads the model weights and configuration from the checkpoint
It returns the model in evaluation mode ready for inference
2. Generate Predictions: First ensures CPU is being used, then loads IMU data from a tensor file, 
processes it to match model's input format uses it for 3d body pose estimation

"""
def load_model(checkpoint_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '' # Disable CUDA explicitly, force all computation to happen on CPU
    device=torch.device('cpu')
    try:
        # Ensure sure checkpoint exists
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError("Checkpoint not found")
        print(f"Loading model from checkpoint: {checkpoint_path}")
        
        # Load checkpoint with CPU mapping
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'hyper_parameters' in checkpoint and 'config' in checkpoint['hyper_parameters']:
            config_dict = checkpoint['hyper_parameters']['config']
            
            # Ceate Config with CPU device
            config = Config(
                experiment=config_dict.experiment if hasattr(config_dict, 'experiment') else "IMUPoserGlobalModel",
                model=config_dict.model if hasattr(config_dict, 'model') else "GlobalModelIMUPoser",
                project_root_dir= BASE_DIR,
                joints_set=config_dict.joints_set if hasattr(config_dict, 'joints_set') else amass_combos['global'],
                normalize=False,
                r6d=True,
                loss_type="mse",
                use_joint_loss=True,
                device='cpu',
                og_smpl_model_path=  os.path.join(BASE_DIR, "src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")
            )
        else:
            config = Config(
                experiment="IMUPoserGlobalModel",
                model="GlobalModelIMUPoser",
                project_root_dir=BASE_DIR,
                joints_set=amass_combos['global'],
                normalize=False,
                r6d=True,
                loss_type="mse",
                use_joint_loss=True,
                device='cpu',
                og_smpl_model_path=  os.path.join(BASE_DIR, "src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")
            )
        
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




    
def generate_prediction(model, input_tensor_path,output_path=None):
    device = torch.device('cpu') # Disable CUDA explicitly, force all computation to happen on CPU
    input_tensor_path = Path(input_tensor_path) # Load input tensor
    print(f"Loading input tensor from: {input_tensor_path}")
    input_data = torch.load(input_tensor_path, map_location=device) # Load tensor with CPU mapping
    
    if isinstance(input_data, dict) and 'imu_data' in input_data:
        input_tensor = input_data['imu_data'] # Extract IMU data from dictionary
    else:
        input_tensor = input_data
    
    # Tenson is on CPU and correct type
    if isinstance(input_tensor, torch.Tensor):
        input_tensor = input_tensor.to(device).float()
    else:
        raise TypeError("Input data is not a tensor")
    
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Prepare input for model (model expects [batch_size, seq_length, features])
    if len(input_tensor.shape) == 2: #
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension at position 0
    
    # Get sequence length for each batch
    seq_length = input_tensor.shape[1]
    input_lengths = torch.tensor([seq_length], device=device, dtype=torch.long)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad(): # Disable gradient calculation to reduce memory usage
        # Ensure model is in evaluation mode and on CPU
        model.eval()
        model = model.to(device)
        predictions = model(input_tensor, input_lengths)
    
    # Extract pose predictions from model output
    if isinstance(predictions, tuple) and len(predictions) >= 1:
        pose_predictions = predictions[0]
    else:
        pose_predictions = predictions
    
    # Ensure predictions are on CPU and float type
    pose_predictions = pose_predictions.to(device).float() # type: ignore
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