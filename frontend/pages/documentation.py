import streamlit as st # type: ignore



################## DOCUMENTATION HELP PAGE ###################
def documentation_help_page():
    """Render the Documentation/Help page"""
    st.title("Documentation & Help")
    
    st.markdown("""
    ## Welcome to TEMPO's Help & Documentation

    This page provides detailed guidance on how to use the TEMPO application, troubleshoot common issues, and answers to frequently asked questions (FAQs).

    ### How to Use TEMPO
    TEMPO integrates wearable sensors for real-time motion tracking and 3D pose estimation. Follow the steps below to get started:

    1. **Set up your sensors**: Connect your wearable devices, such as wrist sensors, phone sensors, and eSense earbuds, by following the setup instructions on the "Data Collection Hub" page.
    
    2. **Data Collection**: Once your sensors are connected, you can begin recording motion data. The app will show you progress on each sensor’s setup, guiding you step by step.
    
    3. **Data Analysis**: After collecting your data, use the "Data Analysis" page to upload raw data files, process them, and generate 3D pose visualisations.
    
    4. **Results & Reports**: View your analysis results in real-time and export them for further review. You can also compare results across multiple sessions.

    ### Frequently Asked Questions (FAQs)
    **Q: What sensors are supported by TEMPO?**
    A: TEMPO supports a range of sensors, including:
    - Wrist-mounted accelerometers and gyroscopes (MetaWear)
    - Phone sensors (using the Physics Toolbox Sensor Suite app)
    - eSense earbuds for additional data collection

    **Q: How do I connect my sensors to the TEMPO app?**
    A: To connect your sensors, follow the setup steps in the "Data Collection Hub". Ensure that Bluetooth is enabled, and your sensors are correctly configured before starting data collection.

    **Q: My sensors are not showing up in the app, what should I do?**
    A: If your sensors are not detected, try the following:
    - Ensure Bluetooth is enabled on your phone or computer.
    - Double-check that your devices are powered on and within range.
    - Restart the app and reconnect the devices.
    
    **Q: How do I process the data after uploading it?**
    A: After uploading your sensor data on the "Data Analysis" page, you can click the "Process Files" button to initiate the processing pipeline. This will analyze the motion data and generate reports with visualisations.

    **Q: How can I export the analysis results?**
    A: You can export your analysis results in CSV or PDF format by clicking the "Export" button on the results page. Detailed instructions on how to save and share your reports will be displayed.

    ### Troubleshooting Tips
    - **No video output**: Ensure the video file is in MP4 format and placed in the correct directory.
    - **Slow processing time**: Processing large datasets may take time. Ensure that your device meets the recommended system requirements.
    - **Data quality issues**: If you're seeing poor-quality data, make sure your sensors are calibrated correctly and sampling rates are set to 50Hz.


    """, unsafe_allow_html=True)