import streamlit as st

### CREATE SENSOR SECTION: Interactive data collection hub for managing devices. Each device has its own expandable section with 
# step-by-step setup instructions and progress tracking 
# 1. Phone/Watch Expansion: Creates a collapsible section for device setup with two steps, "Install App" and "Configure Settings"
# Installation provides app store links and configurations shows how to use the apps 
# There is also a button to start from beginning 
# 2. eSense Earbuds Expander Section: Defines "Web App Setup" and "Link to Data Collection" steps, providing 
# context for the web application and linking to it.

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
        steps = ["Install App", "Configure Settings"]
        current_step = st.session_state.phone_step
        
        # Progress indicator to see completion
        if current_step < len(steps):
            st.progress(current_step / (len(steps) - 1))
            st.write(f"Current Step: {steps[current_step]}")
        
        #  Installation step: showing application download links 
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
            
            if st.button("✅ Mark Installation Complete", key="phone_install_complete"):
                st.session_state.phone_step = 1
                st.experimental_rerun()
        
        # Configuration step of how touse the app
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
        steps = ["Install App", "Configure Sensors"]  #steps for progress bar again
        current_step = st.session_state.wrist_step
        
        # Show progress indicator if not completed
        if current_step < len(steps):
            st.write(f"Current Step: {steps[current_step]}")
        
        # Installation steps and advice 
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
            if st.button("🔄 Start from Beginning"): # start over button
                st.session_state.wrist_step = 0
                st.experimental_rerun()
    
    # eSense Expander
    with st.expander("🎧 eSense Earbuds", expanded=False):
        st.subheader("eSense Earbuds")
        steps = ["Web App Setup", "Link to Data Collection"]
        current_step = st.session_state.esense_step
    
        if current_step < len(steps): #progress tracker
            st.progress(current_step / (len(steps) - 1))
            st.write(f"Current Step: {steps[current_step]}")
        
        # Web App introduction and context
        if current_step == 0:
            st.markdown("""
            ### eSense Data Recording Web App
            
            eSense can record data using our specialised web application:
            
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
        
        # Link to Data Collection for second step
        elif current_step == 1:
            st.markdown("""
            ### Link to Data Collection
            
            You are now ready to proceed to the data collection page:
            
            - Ensure eSense earbuds are paired
            - Check Bluetooth connectivity
            - Prepare your recording environment
            - Find this page on the navigation bar
            """)
    
        
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


