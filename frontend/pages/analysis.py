import streamlit as st # type: ignore
import cv2 # type: ignore
import os
from pathlib import Path
import torch
from frontend.analysis.joints import process_joint_angles
from frontend.analysis.speed import process_movement_speed
from frontend.analysis.accuracy import process_pose_accuracy
from frontend.utils.helpers import process_uploaded_files


import numpy as np
import pandas as pd
base_dir = "/dcs/22/u2254377/cs310/IMUPoser"
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
        data_dir = "rawdata"
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
        st.header("3D Pose visualisation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Section for visualisation frames carousel
            st.subheader("Pose Frames")
            
            # Check if frames directory exists
            frames_dir = os.path.join(base_dir,"/output/pose_frames_dots")
            if os.path.exists(frames_dir):
                # Get all PNG files in the directory
                frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
                
                if frame_files:
                    # Create a custom carousel
                    st.write("Use the slider to navigate through pose frames:")
                    
                    # Frame selection slider
                    selected_frame_idx = st.slider(
                        "Frame", 
                        min_value=0, 
                        max_value=len(frame_files)-1, 
                        value=0,
                        key="frame_slider"
                    )
                    
                    # Display the selected frame
                    selected_frame_path = os.path.join(frames_dir, frame_files[selected_frame_idx])
                    st.image(selected_frame_path, use_column_width=True, caption=f"Frame {selected_frame_idx+1}/{len(frame_files)}")
                    
    
                    # Additional controls
                    with st.expander("Frame Controls", expanded=False):
                        # Option to download the current frame
                        with open(selected_frame_path, "rb") as file:
                            st.download_button(
                                label="Download Current Frame",
                                data=file,
                                file_name=f"pose_frame_{selected_frame_idx}.png",
                                mime="image/png"
                            )
                        
                        # Option to view the GIF
                        gif_path = os.path.join(base_dir, "output/output_frames.gif")
                        if os.path.exists(gif_path):
                            st.write("Full Animation:")
                            st.image(gif_path, use_column_width=True)
                            
                            with open(gif_path, "rb") as file:
                                st.download_button(
                                    label="Download Animation GIF",
                                    data=file,
                                    file_name="pose_animation.gif",
                                    mime="image/gif",
                                    key="gif_download"
                                )
  
        with col2:
            # Section for multi-view grid and analysis
            st.subheader("Multi-View Analysis")
            
            # Display the multi-view grid if available
            grid_path = os.path.join(base_dir, "/output/pose_grid_dots.png")
            if os.path.exists(grid_path):
                st.image(grid_path, use_column_width=True, caption="Multi-view pose grid")
                
                # Add download button for the grid
                with open(grid_path, "rb") as file:
                    st.download_button(
                        label="Download Grid Image",
                        data=file,
                        file_name="pose_grid.png",
                        mime="image/png",
                        key="grid_download"
                    )
            else:
                st.info("Multi-view grid not available. Generate visualisations to create it.")
            
        
        # Analysis section below the two columns
        st.divider()
        st.subheader("Pose Analysis Results")
        
        # Create tabs for different analysis types
        analysis_tabs = st.tabs(["Joint Angles", "Movement Speed", "Pose Accuracy"])
        
        with analysis_tabs[0]:  # Joint Angles
            try:
                # Load predictions
                predictions_path = os.path.join(base_dir, '/rawdata/processed/predictions.pt')
                if os.path.exists(predictions_path):
                    predictions = torch.load(predictions_path)
                    st.write("### Joint Angle Analysis")
                    st.write("The joint angle analysis shows how the angles between connected body segments change over time using the dot product. For example the knee angle is calculated between hip, knee and ankle joint and the elbow joint is calculated between the shoulder, elbow and wrist joint.")

                    # Add debug information
                    if isinstance(predictions, dict) and 'joints' in predictions: 
                        angles = process_joint_angles(predictions)
                        
                        # Create columns for joint angles
                        angle_cols = st.columns(3)
                        col_idx = 0
                        
                        for joint, angle_values in angles.items():
                            with angle_cols[col_idx % 3]:
                                st.metric(
                                    f"{joint.replace('_', ' ').title()}",
                                    f"{np.mean(angle_values):.2f}°",
                                    f"Max: {np.max(angle_values):.2f}°"
                                )
                            col_idx += 1
                        
                        # Plot selected joint angles over time
                        st.write("### Joint Angles Over Time")
                        selected_joints = st.multiselect(
                            "Select joints to display:",
                            options=list(angles.keys()),
                            default=list(angles.keys())[:2]
                        )
                        
                        if selected_joints:
                            angle_data = {joint: angles[joint] for joint in selected_joints}
                            angle_df = pd.DataFrame(angle_data)
                            st.line_chart(angle_df)
                else:
                    st.info("No predictions data found. Please process your data first.")
                    
            except Exception as e:
                st.error(f"Error processing joint angles: {str(e)}")
                st.write("Error details:", str(e))
        
        with analysis_tabs[1]:  # Movement Speed
            try:
                # Load predictions
                predictions_path = os.path.join(base_dir, '/rawdata/processed/predictions.pt')
                if os.path.exists(predictions_path):
                    predictions = torch.load(predictions_path)
                    
                    speeds, stats = process_movement_speed(predictions)
                    
                    # Create chart data
                    speed_data = pd.DataFrame(speeds)
                    
                    # Display the line chart
                    st.write("### Movement Speed Analysis")
                    st.write("The speed graph shows the velocity (in centimeters per second) of different key body parts over time throughout your motion sequence.")
                    
                    # Create columns for movement metrics
                    metric_cols = st.columns(len(stats) if stats else 3)
                    
                    for i, (part, part_stats) in enumerate(stats.items()):
                        with metric_cols[i % len(metric_cols)]:
                            st.metric(
                                f"{part.replace('_', ' ').title()}",
                                f"{part_stats['average']:.2f} cm/s",
                                f"Peak: {part_stats['max']:.2f} cm/s"
                            )
                    
                    # Display chart with options
                    selected_parts = st.multiselect(
                        "Select body parts to display:",
                        options=list(speed_data.columns),
                        default=list(speed_data.columns)[:3]
                    )
                    
                    if selected_parts:
                        filtered_data = speed_data[selected_parts]
                        st.line_chart(filtered_data)
                        
                    st.write("**Interpretation Guide:**")
                    st.write("- **Peaks in the graph:** These represent moments of fast movement for that body part")
                    st.write("- **Valleys or low points:** These show when that body part is moving slowly or is relatively still")
                else:
                    st.info("No predictions data found. Please process your data first.")
                    
            except Exception as e:
                st.error(f"Error processing movement speeds: {str(e)}")
                import traceback
                with st.expander("See detailed error information", expanded=False):
                    st.write("Full error details:", str(e))
                    st.write("Traceback:", traceback.format_exc())
        
        with analysis_tabs[2]:  # Pose Accuracy
            try:
                # Load predictions
                predictions_path = os.path.join(base_dir, '/rawdata/processed/predictions.pt')
                if os.path.exists(predictions_path):
                    predictions = torch.load(predictions_path)
                    
                    # Process pose accuracy
                    metrics, stats = process_pose_accuracy(predictions)
                    
                    # Show statistics
                    st.write("### Accuracy Statistics")
                    
                    # Create columns for each metric
                    cols = st.columns(len(stats) if stats else 3)
                    
                    for i, (metric, metric_stats) in enumerate(stats.items()):
                        with cols[i % len(cols)]:
                            st.metric(
                                label=metric.replace('_', ' ').title(),
                                value=f"{metric_stats['average']:.2f}",
                                delta=f"Range: {metric_stats['min']:.2f} - {metric_stats['max']:.2f}"
                            )
                    
                    # Show time-series data if available
                    if metrics and isinstance(metrics, dict):
                        st.write("### Metrics Over Time")
                        metrics_df = pd.DataFrame(metrics)
                        st.line_chart(metrics_df)
                    
                    # Detailed explanation
                    with st.expander("Metrics Explanation", expanded=False):
                        st.write("""
                        - **Smoothness**: Measures motion smoothness, indicating control and coordination
                        - **Symmetry Score**: Measures left-right body symmetry and can help to detect muscle imbalances 
                        - **Posture Score**: Measures overall posture alignment, this is important as poor posture can lead to back pain and neck strain
                        """)
                else:
                    st.info("No predictions data found. Please process your data first.")
                    
            except Exception as e:
                st.error(f"Error processing pose accuracy: {str(e)}")
                with st.expander("See detailed error information", expanded=False):
                    st.write("Error details:", str(e))
        
        
        with analysis_tabs[1]:  # Movement Speed
            try:
                # Load predictions
                predictions_path = os.path.join(base_dir, '/rawdata/processed/predictions.pt')
                if os.path.exists(predictions_path):
                    predictions = torch.load(predictions_path)
                    
                    speeds, stats = process_movement_speed(predictions)
                    
                    # Create chart data
                    speed_data = pd.DataFrame(speeds)
                    
                    # Display the line chart
                    st.write("### Movement Speed Analysis")
                    st.write("The speed graph shows the velocity (in centimeters per second) of different key body parts over time throughout your motion sequence.")
                    
                    # Create columns for movement metrics
                    metric_cols = st.columns(len(stats) if stats else 3)
                    
                    for i, (part, part_stats) in enumerate(stats.items()):
                        with metric_cols[i % len(metric_cols)]:
                            st.metric(
                                f"{part.replace('_', ' ').title()}",
                                f"{part_stats['average']:.2f} cm/s",
                                f"Peak: {part_stats['max']:.2f} cm/s"
                            )
                    
                    # Display chart with options
                    selected_parts = st.multiselect(
                        "Select body parts to display:",
                        options=list(speed_data.columns),
                        default=list(speed_data.columns)[:3]
                    )
                    
                    if selected_parts:
                        filtered_data = speed_data[selected_parts]
                        st.line_chart(filtered_data)
                        
                    st.write("**Interpretation Guide:**")
                    st.write("- **Peaks in the graph:** These represent moments of fast movement for that body part")
                    st.write("- **Valleys or low points:** These show when that body part is moving slowly or is relatively still")
                else:
                    st.info("No predictions data found. Please process your data first.")
                    
            except Exception as e:
                st.error(f"Error processing movement speeds: {str(e)}")
                import traceback
                with st.expander("See detailed error information", expanded=False):
                    st.write("Full error details:", str(e))
                    st.write("Traceback:", traceback.format_exc())
        
        with analysis_tabs[2]:  # Pose Accuracy
            try:
                # Load predictions
                predictions_path = os.path.join(base_dir, '/rawdata/processed/predictions.pt')
                if os.path.exists(predictions_path):
                    predictions = torch.load(predictions_path)
                    
                    # Process pose accuracy
                    metrics, stats = process_pose_accuracy(predictions)
                    
                    # Show statistics
                    st.write("### Accuracy Statistics")
                    
                    # Create columns for each metric
                    cols = st.columns(len(stats) if stats else 3)
                    
                    for i, (metric, metric_stats) in enumerate(stats.items()):
                        with cols[i % len(cols)]:
                            st.metric(
                                label=metric.replace('_', ' ').title(),
                                value=f"{metric_stats['average']:.2f}",
                                delta=f"Range: {metric_stats['min']:.2f} - {metric_stats['max']:.2f}"
                            )
                    
                    # Show time-series data if available
                    if metrics and isinstance(metrics, dict):
                        st.write("### Metrics Over Time")
                        metrics_df = pd.DataFrame(metrics)
                        st.line_chart(metrics_df)
                    
                    # Detailed explanation
                    with st.expander("Metrics Explanation", expanded=False):
                        st.write("""
                        - **Smoothness**: Measures motion smoothness, indicating control and coordination
                        - **Symmetry Score**: Measures left-right body symmetry and can help to detect muscle imbalances 
                        - **Posture Score**: Measures overall posture alignment, this is important as poor posture can lead to back pain and neck strain
                        """)
                else:
                    st.info("No predictions data found. Please process your data first.")
                    
            except Exception as e:
                st.error(f"Error processing pose accuracy: {str(e)}")
                with st.expander("See detailed error information", expanded=False):
                    st.write("Error details:", str(e))






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

