import streamlit as st # type: ignore
import os
from frontend.utils.device_data_upload import device_data_upload
from frontend.utils.create_sensor_section import create_sensor_section
from frontend.utils.pose_analysis import pose_analysis

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
        "🎥 3D Pose Visualisation",
        "📈 Pose Analysis"
    ])

    with tabs[0]:
        create_sensor_section()

    with tabs[1]:
        device_data_upload(base_dir)

    with tabs[2]:
        st.header("3D Pose visualisation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Section for visualisation frames carousel
            st.subheader("Pose Frames")
            
            # Check if frames directory exists
            frames_dir = os.path.join(base_dir,"output", "pose_frames_dots")
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
                        gif_path = os.path.join(base_dir, "output", "output_frames.gif")
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
            grid_path = os.path.join(base_dir, "output/pose_grid_dots.png")
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

    with tabs[3]: 
        pose_analysis(base_dir)

        
        