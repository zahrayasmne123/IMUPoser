import streamlit as st # type: ignore
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frontend.pages.documentation import documentation_help_page
from frontend.pages.analysis import data_analysis_page
from frontend.pages.about import about_us_page
from frontend.pages.esens_collection import esens_collection_page


# Set page configuration
st.set_page_config(
    page_title="TEMPO - Medical Motion Tracking",
    page_icon="🏥",
    layout="wide"
)


###### APP.PY: Main hub for front end display with home page, data analysis section, documentation, 
# and about us section #######

# 1. Set Page Config:  Sets up the page with title and wide layour
# 2. Colour Scheme: CSS Implementations for whole website styling
# 3. Display TEMPO Title: Creates gradient title "TEMPO" and subtitle using custom CSS
# 4. Home Page: Main landing page with a statistics row showing four metrics, introductory welcome message for TEMPO
# key features, quick start guide and button to start data collection
# 5. Main: Function that runs the Streamlit app and creates sidebar navigation to load other pages

#color scheme
main_colour = "#0066cc"
secondary_colour = "#ff9900"

# CSS Implementation
st.markdown("""
    <style>
    /* Base styles */
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
    }

    /* Sophisticated header styles */
    .header-modern {
        background-color: white;
        padding: 2rem 3rem;
        margin: -4rem -4rem 2rem -4rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    </style>
    """, unsafe_allow_html=True)



def display_tempo_title():
    """
    Displa title for the TEMPO application
    """
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
    
    .tempo-subtitle {
        font-family: 'Arial', sans-serif;
        font-size: 1.5em;
        color: #34495e;
        text-align: center;
        margin-bottom: 30px;
        font-style: italic;
    }
    </style>
    
    <div class="tempo-title">TEMPO</div>
    <div class="tempo-subtitle">Tracking and Estimating Motion for Patient Observation</div>
    """, unsafe_allow_html=True)

########################### HOME PAGE #############################

def home_page():
    display_tempo_title()
    st.markdown("""
        <style>
        .highlight-container {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #0066cc;
            margin: 10px 0;
        }
        
        .stat-box {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .feature-card {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            height: 100%;
            transition: transform 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
        }
        
        .centered-image {
            display: block;
            margin: auto;
            max-width: 100%;
            height: auto;
        }
        
        .gradient-text {
            background: linear-gradient(90deg, #0066cc, #00cc99);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    # Metric Statistics Row
    metric1, metric2, metric3, metric4 = st.columns(4)
    
    with metric1:
        st.markdown("""
            <div class="stat-box">
                <h3>50Hz</h3>
                <p>Sampling Rate</p>
            </div>
        """, unsafe_allow_html=True)


    with metric2:
        st.markdown("""
            <div class="stat-box">
                <h3>70%</h3>
                <p>Motion Accuracy</p>
            </div>
        """, unsafe_allow_html=True)

    with metric3:
        st.markdown("""
            <div class="stat-box">
                <h3>4</h3>
                <p>Motion Sensors</p>
            </div>
        """, unsafe_allow_html=True)

    with metric4:
        st.markdown("""
            <div class="stat-box">
                <h3>17</h3>
                <p>Tracked Points</p>
            </div>
        """, unsafe_allow_html=True)

    # introductory message
    st.markdown("""
        <div class="highlight-container">
            <h2>Welcome to TEMPO</h2>
            <p style="font-size: 1.1em; line-height: 1.6;">
                TEMPO was build to change how motion tracking is done in medical applications. It combines 
                acessible wearable technology with advanced 3D pose estimation. Whether you're 
                a medical professional, researcher, or healthcare provider, TEMPO offers 
                useful tools for accurate motion analysis.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Main Features Section
    st.markdown("## 🚀 Key Features", unsafe_allow_html=True)
    
    key_feature1, key_feature2 = st.columns(2)
    
    with key_feature1:
        st.markdown("""
            <div class="feature-card">
                <h3>📊 Motion Tracking</h3>
                <ul>
                    <li>High-precision sensor data collection</li>
                    <li>Multi-device synchronisation</li>
                    <li>Data visualisation</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
    with key_feature2:
        st.markdown("""
            <div class="feature-card">
                <h3>🔍 Data Analysis</h3>
                <ul>
                    <li>Advanced signal processing</li>
                    <li>Statistical analysis tools</li>
                    <li>Export capabilities</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        

    # Quick Start Guide
    st.markdown("## 🚀 Quick Start Guide")
    
    quick_start_guide = st.tabs(["1. Setup Devices", "2. Record Data", "3. Analyze Results", "4. Generate Reports"])
    
    with quick_start_guide[0]:
        st.markdown("""
            ### Setting Up Your Devices
            - Connect your wearable sensors
            - Configure sampling rates
            - Verify connections
            - Calibrate devices
        """)
    
    with quick_start_guide[1]:
        st.markdown("""
            ### Recording Motion Data
            - Position sensors correctly
            - Start synchronised recording
            - Save recorded sessions
        """)
    
    with quick_start_guide[2]:
        st.markdown("""
            ### Analysing Your Results
            - Process raw data
            - Generate 3D visualisations
            - Review motion patterns
        """)
    
    with quick_start_guide[3]:
        st.markdown("""
            ### Generating Reports
            - Create detailed summaries
            - Export visualisations
        """)

    # Call-to-Action Section
    st.markdown("""
        <div style="text-align: center; padding: 40px 0;">
            <h2>Ready to Get Started?</h2>
            <p style="font-size: 1.2em; margin: 20px 0;">
                Begin your journey with TEMPO by setting up your first device.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Action Buttons
    metric1, metric2, metric3 = st.columns([1,2,1])
    with metric2:
        if st.button("🎯 Start Data Collection", use_container_width=True):
            st.info("Please select 'Data Analysis' from the sidebar and go to the 'Collect Device Data' tab.")
    



def main():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'main'

    st.sidebar.title("Navigation") # Create sidebar navigation

    page = st.sidebar.radio(
        "Go to",
        ["Home", "Data Analysis", "About Us", "Documentation & Help","eSens Collection Page" ],
        key="nav_radio"
    )

    # Display selected page from the sidebar
    if page == "Data Analysis":
        data_analysis_page()
    elif page == "Home":
        home_page()
    elif page == "About Us":
        about_us_page()
    elif page == "eSens Collection Page":
        esens_collection_page()
    else:
        documentation_help_page()

if __name__ == "__main__":
    main()

