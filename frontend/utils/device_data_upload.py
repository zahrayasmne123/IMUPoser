import streamlit as st 
import os
import io
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from frontend.utils.helpers import process_uploaded_files

################ DEVICE DATA UPLOAD ################
# This page provides two upload methods: local file upload and Google Drive integration. It detects device types from filenames,
#  validates the data a device type and processes the uploaded files for 3D Pose Estimation
# 1. Initalise Google Cloud Connection: Initialises the Google Drive API servic using service account stored in Streamlit secrets
# 2. Device Data Upload: header and data directory for raw data storage
# 3. File Processing: Combines files from both local uploads and Google Drive
# 4. Device Status Display: Shows a visual summary of connected devices, watches need gyroscope and accelerometer data
# 5. Data Processin: Data processing is only enabled when there is at elast one full device (ie no missing accel). Then
# trained IMUPoser 3d model is ran as a black box using uploaded files to make 3d pose estimation predictions.
# 6. On success, display tensor size, prediction generated and next steps
#
def initialise_google_cloud_connection():
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    return build('drive', 'v3', credentials=credentials)

def device_data_upload(data_dir):
    st.header("Device Data Upload")
    data_dir = "rawdata"
    os.makedirs(data_dir, exist_ok=True)
    
    # Show informational message about flexible uploads
    st.info("""
        **Upload data from any available devices.** 
        
        The system will automatically detect device types based on file name patterns.
        You can upload any combination of files - processing works with any available data.
    """)
    
    upload_tabs = st.tabs(["Local Upload", "Google Drive"])  # tabs for different upload methods
    
    with upload_tabs[0]:
        # Local file upload
        uploaded_files = st.file_uploader(
            "Upload IMU data files",
            type=['csv'],
            accept_multiple_files=True,
            help="Upload any available IMU data CSV files (phone, watch, earbuds)"
        )
    
    with upload_tabs[1]:
        # Google Drive integration
        FOLDER_ID = "15Hmxt55CN3HitGArGWKcRApT8P4qDrHM"
        drive_files = []
        
        try:
            service = initialise_google_cloud_connection()
            with st.spinner("Loading files from Google Drive..."):  # Get files from Google Drive with spiner 
                results = service.files().list( 
                    q=f"'{FOLDER_ID}' in parents",
                    pageSize=20,
                    fields="nextPageToken, files(id, name, modifiedTime)"
                ).execute()
                drive_files = results.get('files', [])

            
            #subfunction to sort filesby modification date
            def get_modified_time(file):
                return file['modifiedTime']


            files_sorted = sorted(drive_files, key=get_modified_time, reverse=True)
  
            
            # File selection with more details
            st.write("### Select Data Files")
            st.write("Choose files from Google Drive to import. Files are sorted by most recently modified.")
            
            # Convert file list to options with readable dates
            file_options = []
            for i, file in enumerate(files_sorted):
                modified_time = pd.to_datetime(file['modifiedTime']).strftime("%Y-%m-%d %H:%M")
                file_options.append(f"{file['name']} (Modified: {modified_time})")
            
            # Allow multi-selection
            selected_files_indices = []
            selected_file_options = st.multiselect(
                "Available files:",
                options=file_options
            )
            
            for option in selected_file_options:
                selected_files_indices.append(file_options.index(option))
            
            selected_drive_files = [files_sorted[idx] for idx in selected_files_indices]
            
            # Button to import selected files
            if selected_drive_files and st.button("Import Selected Files", key="import_drive_button"):
                drive_uploaded_files = []
                
                #saving files from drive to loval sotrage 
                with st.spinner("Importing files from Google Drive..."):
                    for file in selected_drive_files:
                        request = service.files().get_media(fileId=file['id'])
                        fh = io.BytesIO()
                        downloader = MediaIoBaseDownload(fh, request)
                        done = False
                        
                        while not done:
                            _, done = downloader.next_chunk()
                        
                        # Save the file locally
                        fh.seek(0)
                        file_path = os.path.join(data_dir, file['name'])
                        with open(file_path, "wb") as f:
                            f.write(fh.getvalue())
                        
                        # Create a file-like object for processing
                        file_obj = type('obj', (object,), {
                            'name': file['name'],
                            'getvalue': lambda: fh.getvalue()
                        })
                        drive_uploaded_files.append(file_obj)
                    
                    st.success(f"Successfully imported {len(drive_uploaded_files)} files from Google Drive")
                    
                    # Store in session state so they can be processed
                    if 'drive_uploaded_files' not in st.session_state:
                        st.session_state.drive_uploaded_files = []
                    st.session_state.drive_uploaded_files = drive_uploaded_files
    
        except Exception as e:
            st.error(f"Error accessing Google Drive: {str(e)}")
    
    # combine local uploaded files with imported Drive files
    all_uploaded_files = []
    if 'uploaded_files' in locals() and uploaded_files:
        all_uploaded_files.extend(uploaded_files)
    if 'drive_uploaded_files' in st.session_state and st.session_state.drive_uploaded_files:
        all_uploaded_files.extend(st.session_state.drive_uploaded_files)
    
    # process all files
    if all_uploaded_files:
        st.markdown("### Uploaded Files")
        
        # helper to detect device type from filename
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
                return "Phone"  # default to phone if no other matches
        
        # table to display detected file types
        file_data = []
        for file in all_uploaded_files:
            detected_type = detect_device_type(file.name)
            file_path = os.path.join(data_dir, file.name)
            
            # ensure file is saved if not alr (errror andling)
            if not hasattr(file, '_file_path_saved'):
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())
                
            # Add to file data
            file_size = len(file.getvalue()) if hasattr(file, 'getvalue') else os.path.getsize(file_path)
            file_data.append({
                "Filename": file.name,
                "Detected Device": detected_type,
                "Size": f"{round(file_size / 1024, 2)} KB",
                "Source": "Google Drive" if hasattr(file, '_from_drive') else "Local Upload"
            })
        
        st.table(file_data) #display data 
         
        # check which device types are complete
        device_status = {
            "Phone": False,
            "Left Watch": False,
            "Right Watch": False,
            "Earbuds": False
        }
        detected_types = [item["Detected Device"] for item in file_data]
        
        # phone data
        if any("Phone" in device for device in detected_types):
            device_status["Phone"] = True
        
        # earbuds data
        if any("Earbuds" in device for device in detected_types):
            device_status["Earbuds"] = True
        
        # left watch (needs both accelerometer and gyroscope)
        if any("Left Watch (Accelerometer)" in device for device in detected_types) and \
        any("Left Watch (Gyroscope)" in device for device in detected_types):
            device_status["Left Watch"] = True
        
        # right watch (needs both accelerometer and gyroscope)
        if any("Right Watch (Accelerometer)" in device for device in detected_types) and \
        any("Right Watch (Gyroscope)" in device for device in detected_types):
            device_status["Right Watch"] = True
            

        st.markdown("### Device Status") # display device status clearly 
        device_cols = st.columns(4)
        for i, (device, status) in enumerate(device_status.items()):
            with device_cols[i]:
                if status:
                    st.success(f"✅ {device}: Complete")
                else:
                    # for watches, provide more detailed feedback
                    if device == "Left Watch" and any("Left Watch" in d for d in detected_types):
                        missing = "Gyroscope" if "Left Watch (Accelerometer)" in detected_types else "Accelerometer"
                        st.warning(f"⚠️ {device}: Missing {missing}")
                    elif device == "Right Watch" and any("Right Watch" in d for d in detected_types):
                        missing = "Gyroscope" if "Right Watch (Accelerometer)" in detected_types else "Accelerometer"
                        st.warning(f"⚠️ {device}: Missing {missing}")
                    else:
                        st.info(f"ℹ️ {device}: No data")
        
        # only enable processing if at least one device has complete data
        can_process = any(device_status.values())
    else:
        can_process = False
    
    # process button section
    st.divider()
    
    if not can_process and all_uploaded_files:
        st.warning("Please upload complete data for at least one device to continue. For watches, both accelerometer and gyroscope data are needed.")
    elif not all_uploaded_files:
        st.info("Please upload data files to continue")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        process_button = st.button(
            "Process Files",
            key="process_button",
            disabled=not can_process,
            use_container_width=True
        )
        
    # when processbutton is able to be pressed...
    if process_button:
        with st.spinner("Processing files and running model inference... This may take a few moments"):
            try:
                output_dir = "rawdata/processed"
                os.makedirs(output_dir, exist_ok=True)
                
                # use checkpoint path
                checkpoint_path = "checkpoints/checkpoint.ckpt"
                
                # process the uploaded files and make model prediction
                active_devices, tensor, predictions = process_uploaded_files(
                    data_dir, 
                    output_dir, 
                    run_model=True,
                    checkpoint_path=checkpoint_path
                )
                
                #  sucess message with insight intro device and prediction putput 
                if tensor is not None:
                    st.success("✅ Processing and inference complete!")
                    
                    #  metrics in columns
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
                    
                    #  device list
                    st.write("**Processed Devices:**")
                    device_cols = st.columns(4)
                    for i, device in enumerate(active_devices):
                        with device_cols[i % 4]:
                            st.success(f"✓ {device}")
                    
                    st.divider()
                    st.markdown("""
                    ### Next Steps
                    
                    You can now:
                    1. Go to the "3D Pose Video" tab to analyze the processed data
                    2. Visualize the predicted poses
                    3. Export results for further analysis
                    """)
                    
                    # mark completion
                    st.session_state.processing_complete = True
                    st.session_state.processed_devices = active_devices
                    if predictions is not None:
                        st.session_state.predictions_available = True
            
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
                st.exception(e)
                st.warning("""
                **Troubleshooting Tips:**
                - Check that CSV files are properly formatted
                - Ensure all required columns are present
                - Verify that timestamps are consistent
                - Make sure the model checkpoint exists at the specified path
                """)