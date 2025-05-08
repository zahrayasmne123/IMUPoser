import streamlit as st # type: ignore
import os
from frontend.utils.device_data_upload import device_data_upload
from frontend.utils.create_sensor_section import create_sensor_section
from src.imuposer.config import BASE_DIR
################## DATA ANALYSIS PAGE ############################
# 1. Creates four tab different data analysis functions in sub pages
# 2. On sub-page 1: Call the create_sensor_section function
# 3. On sub-page 2: Import data collection function
# 4. On sub-Page 3: View pose visualisation: two columns, one with  frame-by-frame visualisation and slider, the 
#               other with a  multi-view grid of poses

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

    # Tabs for different sub pages
    sub_page = st.tabs([
        "📱 Collect Device Data",
        "📤 Upload Device Data",
        "🎥 3D Pose Visualisation",
    ])

    with sub_page[0]:
        create_sensor_section()
    with sub_page[1]:
        device_data_upload(BASE_DIR)

    with sub_page[2]:
        st.header("3D Pose visualisation")
        carosel_frames, multiview_grid = st.columns([1, 1])
        
        with carosel_frames:
            st.subheader("Pose Frames") # visualisation carousel
            
            # Gather individual frames from frames directory 
            frames_dir = os.path.join(BASE_DIR,"output", "pose_frames_dots")
            individual_frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
            
            if individual_frame_files:
                st.write("Use the slider to navigate through pose frames:")
                
                # frame selection slider for carosel
                selected_frame_index = st.slider(
                    "Frame", 
                    min_value=0, 
                    max_value=len(individual_frame_files)-1, 
                    value=0,
                    key="frame_slider"
                )
                
                # display frame chosen by sldie
                selected_frame_path = os.path.join(frames_dir, individual_frame_files[selected_frame_index])
                st.image(selected_frame_path, use_column_width=True, caption=f"Frame {selected_frame_index+1}/{len(individual_frame_files)}")
                

                # Exporting controls
                with st.expander("Frame Controls", expanded=False):
                    with open(selected_frame_path, "rb") as file:  # Option to download the current frame
                        st.download_button(
                            label="Download Current Frame",
                            data=file,
                            file_name=f"pose_frame_{selected_frame_index}.png",
                            mime="image/png"
                        )
                    
                    # Download as gif to view the GIF
                    gif_path = os.path.join(BASE_DIR, "output", "output_frames.gif")
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
  
        with multiview_grid: # Section for multi-view grid and analysis
            st.subheader("Multi-View Analysis")
            
            # Display the multi-view grid image from putput dolder 
            multiview_grid_path = os.path.join(BASE_DIR, "output/pose_grid.png")
            if os.path.exists(multiview_grid_path):
                st.image(multiview_grid_path, use_column_width=True, caption="Multi-view pose grid")
                
                # Download button for multi view grid
                with open(multiview_grid_path, "rb") as file:
                    st.download_button(
                        label="Download Grid Image",
                        data=file,
                        file_name="pose_grid.png",
                        mime="image/png",
                        key="grid_download"
                    )
            else:
                st.info("Multi-view grid not available. Generate visualisations to create it.")


        
        