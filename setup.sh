# Create environment
conda create -n "imuposer" python=3.8

# Activate environment
conda activate imuposer

# Install an older numpy first (for chumpy compatibility)
pip install numpy==1.20.0

# Install chumpy
pip install chumpy==0.70

# Now install PyTorch and other packages
conda install pytorch torchvision torchaudio -c pytorch

# Install remaining requirements
pip install pytorch-lightning==2.4.0
pip install tqdm==4.64.0
pip install matplotlib==3.2.2
pip install opencv-python
pip install scipy==1.7.3
pip install wandb==0.12.11
pip install dotmap

# Install the source package
python -m pip install -e src/