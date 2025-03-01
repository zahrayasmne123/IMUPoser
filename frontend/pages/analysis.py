import streamlit as st # type: ignore
import cv2 # type: ignore
import os
from pathlib import Path
import torch
from frontend.analysis.joints import process_joint_angles
from frontend.analysis.speed import process_movement_speed
from frontend.analysis.accuracy import process_pose_accuracy
from process_sensor_data.imuDataPipeline import full_sensor_pipeline
from application.run_inference import run_inference,load_model

import numpy as np
import pandas as pd

############# DATA ANALYSIS PAGE ######################

def data_analysis_page():
    st.markdown("""
        <style>
        .upload-section {
            margin: 1rem 0;
        }
        
        .metric-container {
            background: white;
            padding: 1rem;
            border-radius: 4px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .metric-value {
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
        }
        
        .metric-label {
            color: #666;
            margin-top: 0.5rem;
        }
        
        /* Clean up tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0.5rem 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Display title using home page style
    st.markdown("""
    <style>
    .tempo-title {
        font-family: 'Arial', sans-serif;
        font-size: 3.5em;
        font-weight: bold;
        background: linear-gradient(to right, #2c3e50, #3498db);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 20px;
        padding: 10px;
    }
    
    </style>
    
    <div class="tempo-title">Data Analysis</div>
    """, unsafe_allow_html=True)

    # Create tabs with minimal styling
    tabs = st.tabs([
        "📱 Collect Device Data",
        "📤 Upload Device Data",
        "🎥 3D Pose Video"
    ])

    with tabs[0]:
        create_sensor_section()

    with tabs[1]:
        st.header("Device Data Upload")
        
        # Create data directory
        data_dir = "1.rawdata"
        os.makedirs(data_dir, exist_ok=True)
        
        # Show informational message about flexible uploads
        st.info("""
            **Upload data from any available devices.** 
            
            The system will automatically detect device types based on file name patterns.
            You can upload any combination of files - processing works with any available data.
        """)
        
        # Use a file uploader that accepts multiple files
        uploaded_files = st.file_uploader(
            "Upload IMU data files",
            type=['csv'],
            accept_multiple_files=True,
            help="Upload any available IMU data CSV files (phone, watch, earbuds)"
        )
        
        # Process uploaded files
        if uploaded_files:
            # Display uploaded files with automatic device detection
            st.markdown("### Uploaded Files")
            
            # Helper to detect device type from filename
            def detect_device_type(filename):
                filename_lower = filename.lower()
                
                if "left" in filename_lower and "accelerometer" in filename_lower:
                    return "Left Watch (Accelerometer)"
                elif "left" in filename_lower and "gyroscope" in filename_lower:
                    return "Left Watch (Gyroscope)"
                elif "right" in filename_lower and "accelerometer" in filename_lower:
                    return "Right Watch (Accelerometer)"
                elif "right" in filename_lower and "gyroscope" in filename_lower:
                    return "Right Watch (Gyroscope)"
                elif any(pattern in filename_lower for pattern in ["earbud", "esense", "headphone"]):
                    return "Earbuds"
                else:
                    return "Phone"  # Default to phone if no other matches
            
            # Create a table to display detected file types
            file_data = []
            for file in uploaded_files:
                detected_type = detect_device_type(file.name)
                file_path = os.path.join(data_dir, file.name)
                
                # Save the file
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())
                
                # Add to file data
                file_data.append({
                    "Filename": file.name,
                    "Detected Device": detected_type,
                    "Size": f"{round(len(file.getvalue()) / 1024, 2)} KB"
                })
            
            # Display the file table
            st.table(file_data)
            
            # Check which device types are complete
            device_status = {
                "Phone": False,
                "Left Watch": False,
                "Right Watch": False,
                "Earbuds": False
            }
            
            detected_types = [item["Detected Device"] for item in file_data]
            
            # Check phone data
            if any("Phone" in device for device in detected_types):
                device_status["Phone"] = True
            
            # Check earbuds data
            if any("Earbuds" in device for device in detected_types):
                device_status["Earbuds"] = True
            
            # Check left watch (needs both accelerometer and gyroscope)
            if any("Left Watch (Accelerometer)" in device for device in detected_types) and \
            any("Left Watch (Gyroscope)" in device for device in detected_types):
                device_status["Left Watch"] = True
            
            # Check right watch (needs both accelerometer and gyroscope)
            if any("Right Watch (Accelerometer)" in device for device in detected_types) and \
            any("Right Watch (Gyroscope)" in device for device in detected_types):
                device_status["Right Watch"] = True
                
            # Show device status
            st.markdown("### Device Status")
            
            # Display device status visually
            for device, status in device_status.items():
                if status:
                    st.success(f"✅ {device}: Complete data available")
                else:
                    # For watches, provide more detailed feedback
                    if device == "Left Watch" and any("Left Watch" in d for d in detected_types):
                        missing = "Gyroscope" if "Left Watch (Accelerometer)" in detected_types else "Accelerometer"
                        st.warning(f"⚠️ {device}: Incomplete (missing {missing} data)")
                    elif device == "Right Watch" and any("Right Watch" in d for d in detected_types):
                        missing = "Gyroscope" if "Right Watch (Accelerometer)" in detected_types else "Accelerometer"
                        st.warning(f"⚠️ {device}: Incomplete (missing {missing} data)")
                    else:
                        st.info(f"ℹ️ {device}: No data uploaded")
            
            # Only enable processing if at least one device has complete data
            can_process = any(device_status.values())
        else:
            can_process = False
        
        # Process button section
        st.divider()
        
        if not can_process and uploaded_files:
            st.warning("Please upload complete data for at least one device to continue. For watches, both accelerometer and gyroscope data are needed.")
        elif not uploaded_files:
            st.info("Please upload data files to continue")
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            process_button = st.button(
                "Process Files",
                key="process_button",
                disabled=not can_process,
                use_container_width=True
            )
            
            # Process button section
        # Process button section
        if process_button:
            with st.spinner("Processing files and running model inference... This may take a few moments"):
                try:
                    # Create output directory
                    output_dir = "rawdata/processed"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Use a default checkpoint path - update this to your actual path
                    checkpoint_path = "checkpoints/checkpoint.ckpt"
                    
                    # Process the uploaded files and run inference
                    active_devices, tensor, predictions = process_uploaded_files(
                        data_dir, 
                        output_dir, 
                        run_model=True,
                        checkpoint_path=checkpoint_path
                    )
                    
                    # Show detailed success message
                    if tensor is not None:
                        st.success("✅ Processing and inference complete!")
                        
                        # Display metrics in columns
                        metrics_cols = st.columns(3)
                        with metrics_cols[0]:
                            st.metric("Active Devices", len(active_devices))
                        with metrics_cols[1]:
                            st.metric("Tensor Shape", f"{tensor.shape[0]} × {tensor.shape[1]}")
                        with metrics_cols[2]:
                            if predictions is not None:
                                st.metric("Predictions", "✓ Generated")
                            else:
                                st.metric("Predictions", "✗ Not available")
                        
                        # Display device list
                        st.write("**Processed Devices:**")
                        device_cols = st.columns(4)
                        for i, device in enumerate(active_devices):
                            with device_cols[i % 4]:
                                st.success(f"✓ {device}")
                        
                        # Next steps
                        st.divider()
                        st.markdown("""
                        ### Next Steps
                        
                        You can now:
                        1. Go to the "3D Pose Video" tab to analyze the processed data
                        2. Visualize the predicted poses
                        3. Export results for further analysis
                        """)
                        
                        # Set session state to mark completion
                        st.session_state.processing_complete = True
                        st.session_state.processed_devices = active_devices
                        if predictions is not None:
                            st.session_state.predictions_available = True
                
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
                    st.exception(e)
                    
                    # Show troubleshooting tips
                    st.warning("""
                    **Troubleshooting Tips:**
                    - Check that CSV files are properly formatted
                    - Ensure all required columns are present
                    - Verify that timestamps are consistent
                    - Make sure the model checkpoint exists at the specified path
                    """)

    with tabs[2]:
        st.header("3D Pose Video Analysis")
        
        video_files = sorted(list(Path('.').glob('*.mp4')))
        if not video_files:
            st.warning("No MP4 files found in the current directory.")
        else:
            selected_video = st.selectbox(
                "Select video file:",
                options=video_files,
                format_func=lambda x: x.name
            )
            
            if selected_video:
                try:
                    # Create two columns for video and stats
                    video_col, stats_col = st.columns([1, 1])
                    
                    with video_col:
                        # Video display
                        st.video(str(selected_video))
                    
                    with stats_col:
                        # Get video information
                        video = cv2.VideoCapture(str(selected_video))
                        fps = video.get(cv2.CAP_PROP_FPS)
                        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = frame_count/fps
                        
                        st.subheader("Video Statistics")
                        
                        # Display metrics vertically in the stats column
                        st.metric("FPS", f"{fps:.2f}")
                        st.metric("Total Frames", frame_count)
                        st.metric("Duration", f"{duration:.2f} seconds")
                        
                        # Additional information
                        st.divider()
                        st.subheader("Analysis Results")
                        st.info("Select options below to analyze the video:")
                        
                        analysis_type = st.selectbox(
                            "Choose Analysis Type",
                            ["Joint Angles", "Movement Speed", "Pose Accuracy"]
                        )
                        
                        if analysis_type == "Joint Angles":
                            try:
                                # Load predictions
                                predictions = torch.load('rawdata/processed_dataset/predictions.pt')
                                st.write("### Joint Angle Analysis")
                                st.write("The joint angle analysis shows how the angles between connected body segments change over time using the dot product. For example the knee angle is calculated between hip, knee and ankle joint and the elbow joint is calculated between the shoulder, elbow and wrist joint.")


                                # Add debug information
                                if isinstance(predictions, dict) and 'joints' in predictions: 
                                    angles = process_joint_angles(predictions)
                                    for joint, angle_values in angles.items():
                                        st.write(f"**{joint.replace('_', ' ').title()}**")
                                        st.write(f"- Average angle: {np.mean(angle_values):.2f}°")
                                        
                            except Exception as e:
                                st.error(f"Error processing joint angles: {str(e)}")
                                # Print more detailed error information
                                st.write("Error details:", str(e))
                        elif analysis_type == "Movement Speed":
                            try:
                                # Load predictions with proper error handling
                                predictions = torch.load('rawdata/processed_dataset/predictions.pt')

                                speeds, stats = process_movement_speed(predictions)
                                
                                # Create chart data
                                speed_data = pd.DataFrame(speeds)
                                
                                # Display the line chart
                                st.write("### Movement Speed Analysis")
                                st.write("The speed graph shows the velocity (in centimeters per second) of different key body parts over time throughout your motion sequence.")
                                st.write("Peaks in the graph: These represent moments of fast movement for that body part")
                                st.write("Valleys or low points: These show when that body part is moving slowly or is relatively still")
                            
                                st.line_chart(speed_data)
                                

                                        
                            except Exception as e:
                                st.error(f"Error processing movement speeds: {str(e)}")
                                st.write("Full error details:", str(e))
                                import traceback
                                st.write("Traceback:", traceback.format_exc())

                        elif analysis_type == "Pose Accuracy":
                            try:
                                # Load predictions
                                predictions = torch.load('rawdata/processed_dataset/predictions.pt')
                                
                                # Process pose accuracy
                                metrics, stats = process_pose_accuracy(predictions)
                                
                            
                                # Show statistics
                                st.write("### Accuracy Statistics")
                                
                                # Create columns for each metric
                                cols = st.columns(len(stats))
                                
                                for i, (metric, metric_stats) in enumerate(stats.items()):
                                    with cols[i]:
                                        st.metric(
                                            label=metric.replace('_', ' ').title(),
                                            value=f"{metric_stats['average']:.2f}",
                                            delta=f"Range: {metric_stats['min']:.2f} - {metric_stats['max']:.2f}"
                                        )
                                
                                # Detailed explanation
                                st.write("### Metrics Explanation")
                                st.write("""
                                - **Smoothness**: Measures motion smoothness, indicating control and coordination
                                - **Symmetry Score**: Measures left-right body symmetry and can help to detect muscle imbalances 
                                - **Posture Score**: Measures overall posture alignment, this is important as poor posture can lead to back pain and neck strain
                                """)
                                
                            except Exception as e:
                                st.error(f"Error processing pose accuracy: {str(e)}")
                                st.write("Error details:", str(e))

                            



                                
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                    st.info("Please ensure the video file is not corrupted and is a valid MP4 format.")




def create_sensor_section():
    st.header("Data Collection Hub")
    st.write("Connect and manage your sensor devices in one place. Follow the guided steps for each device.")
    
    #Session state for tracking progress initialised 
    if 'wrist_step' not in st.session_state:
        st.session_state.wrist_step = 0
    if 'phone_step' not in st.session_state:
        st.session_state.phone_step = 0
    if 'esense_step' not in st.session_state:
        st.session_state.esense_step = 0
    
    # Phone Sensors Expander
    with st.expander("📱 Phone Sensors", expanded=False):
        st.subheader("Phone Sensors")
        # Define the steps for phone setup
        steps = ["Install App", "Configure Settings"]
        current_step = st.session_state.phone_step
        
        # Show progress indicator if not completed
        if current_step < len(steps):
            st.progress(current_step / (len(steps) - 1))
            st.write(f"Current Step: {steps[current_step]}")
        
        # Step 1: Installation
        if current_step == 0:
            st.markdown("""
            ### Download Physics Toolbox Sensor Suite
            
            Choose your device's app store and install the application:
            
            - [🤖 Google Play Store](https://play.google.com/store/apps/details?id=com.chrystianvieyra.physicstoolboxsuite&hl=en_GB)
            - [📱 iOS App Store](https://apps.apple.com/us/app/physics-toolbox-sensor-suite/id1128914250)
            
            #### Installation Tips:
            - Ensure you have a stable internet connection
            - Allow all required permissions during installation
            - Check that your device meets the minimum requirements
            - Make sure you have enough storage space
            """)
            
            # Added key "phone_install_complete"
            if st.button("✅ Mark Installation Complete", key="phone_install_complete"):
                st.session_state.phone_step = 1
                st.experimental_rerun()
        
        # Step 2: Configuration
        elif current_step == 1:
            st.markdown("""
            ### Configure Your Phone Sensors
            
            1. Open Physics Toolbox Sensor Suite
            2. Configure the following settings:
            - Set sampling rate to 50Hz
            - Enable accelerometer and gyroscope
            - Verify sensors are working correctly
            3. Test the recording function
            4. Ensure CSV export is working properly
            """)
            
            # Added key "phone_config_complete"
            if st.button("✅ Configuration Complete", key="phone_config_complete"):
                st.session_state.phone_step = 2
                st.experimental_rerun()
        
        # Completion Card
        elif current_step == 2:
            st.success("🎉 Phone Setup Complete!")
            st.info("""
            You have successfully:
            - Installed Physics Toolbox Sensor Suite
            - Configured all necessary sensor settings
            - Verified the recording and export functionality
            
            Your phone is now ready for data collection!
            """)
            
            # Added key "phone_start_over"
            if st.button("🔄 Start from Beginning", key="phone_start_over"):
                st.session_state.phone_step = 0
                st.experimental_rerun() 
    
    # Wrist Sensors Expander
    with st.expander("⌚ Wrist Sensors", expanded=False):
        st.subheader("MetaSens Wrist Sensors")
        
        # Create progress tracking with only two steps
        steps = ["Install App", "Configure Sensors"]
        current_step = st.session_state.wrist_step
        
        # Show progress indicator if not completed
        if current_step < len(steps):
            st.write(f"Current Step: {steps[current_step]}")
        
        # Step content 
        if current_step == 0:
            st.markdown("""
            ### Getting Started
            Download the MetaWear app for your device:
            
            - [📱 iOS App Store](https://apps.apple.com/us/app/metawear/id1547334547)
            - [🤖 Google Play Store](https://play.google.com/store/apps/details?id=com.mbientlab.metawear.app)
            
            #### Installation Tips:
            - Ensure Bluetooth is enabled on your device
            - Allow necessary permissions when prompted
            - Check for minimum OS requirements
            """)
            
            if st.button("✅ Mark Installation Complete"):
                st.session_state.wrist_step = 1
                st.experimental_rerun()
                
        elif current_step == 1:
            st.markdown("""
            ### Configure Your Sensors
            
            1. Open the MetaWear app
            2. Set sampling rates:
            - Accelerometer: 50Hz
            - Gyroscope: 50Hz
            3. Verify connection status
            4. Download csv files 
            """)
            
            if st.button("✅ Configuration Complete"):
                st.session_state.wrist_step = 2
                st.experimental_rerun()
        
        # Show completion card when all steps are done
        elif current_step == 2:
            st.success("🎉 Setup Complete!")
            st.info("You have successfully set up the wrist sensors and configured all necessary parameters.")
            
            # Add the start over button
            if st.button("🔄 Start from Beginning"):
                st.session_state.wrist_step = 0
                st.experimental_rerun()
    
    # eSense Expander
    with st.expander("🎧 eSense Earbuds", expanded=False):
        st.subheader("eSense Earbuds")
        
        # Create progress tracking with steps
        steps = ["Web App Setup", "Link to Data Collection"]
        current_step = st.session_state.esense_step
        
        # Show progress indicator if not completed
        if current_step < len(steps):
            st.progress(current_step / (len(steps) - 1))
            st.write(f"Current Step: {steps[current_step]}")
        
        # Step 1: Web App Introduction
        if current_step == 0:
            st.markdown("""
            ### eSense Data Recording Web App
            
            eSense can record data using our specialized web application:
            
            #### Key Features:
            - Real-time sensor data collection
            - Compatible with eSense earbuds
            - Seamless data export
            
            #### Getting Started:
            1. Ensure your eSense earbuds are charged
            2. Have Bluetooth enabled on your device
            3. Prepare for data collection
            """)
            
            if st.button("✅ Web App Setup Complete"):
                st.session_state.esense_step = 1
                st.experimental_rerun()
        
        # Step 2: Link to Data Collection
        elif current_step == 1:
            st.markdown("""
            ### Link to Data Collection
            
            You are now ready to proceed to the data collection page:
            
            - Ensure eSense earbuds are paired
            - Check Bluetooth connectivity
            - Prepare your recording environment
            """)
            
            if st.button("🎧 Launch eSense Data Collection Web Application"):
            # Store the page we want to show in session state
                if 'current_page' not in st.session_state:
                    st.session_state.current_page = 'main'
                    
                    st.session_state.current_page = 'esens_collection'
                    st.session_state.esense_step = 2
                    st.success("Redirecting to Data Collection Page...")
                    st.experimental_rerun()
        
        # Completion Card
        elif current_step == 2:
            st.success("🎉 eSense Setup Complete!")
            st.info("""
            You have successfully:
            - Set up the eSense Web App
            - Prepared for data collection
            
            Your eSense earbuds are ready for recording!
            """)
            
            if st.button("🔄 Start from Beginning"):
                st.session_state.esense_step = 0
                st.experimental_rerun()



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
    
    # Run model inference if requested
    predictions = None
    if run_model:
        st.text("Step 2: Running model inference...")
        # Use default checkpoint path if none provided
        if checkpoint_path is None:
            # Update this path to your default checkpoint location
            checkpoint_path = "checkpoints/checkpoint.ckpt"
        
        try:
            # Check if checkpoint exists
            if not os.path.exists(checkpoint_path):
                st.warning(f"Checkpoint not found at {checkpoint_path}. Skipping inference.")
            else:
                # Load the model
                model = load_model(checkpoint_path, device='cpu')
                
                # Run inference
                predictions_path = os.path.join(output_dir, 'predictions.pt')
                predictions = run_inference(model, tensor_path, predictions_path, device='cpu')
                
                st.text(f"✓ Saved predictions to: {predictions_path}")
        except Exception as e:
            st.error(f"Error during model inference: {str(e)}")
            st.text("✗ Model inference failed. Continuing with other processing steps.")
    
    return readable_device_names, tensor, predictions