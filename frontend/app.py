import streamlit as st # type: ignore
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frontend.pages.documentation import documentation_help_page
from frontend.pages.analysis import data_analysis_page
from frontend.pages.about import about_us_page
from frontend.pages.esens_collectionold import render_esense_data_collection

# Set page configuration
st.set_page_config(
    page_title="TEMPO - Medical Motion Tracking",
    page_icon="🏥",
    layout="wide"
)


# Define color scheme
PRIMARY_COLOR = "#0066cc"
SECONDARY_COLOR = "#ff9900"
BACKGROUND_COLOR = "#f0f2f6"

# Custom CSS with multiple header style options
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
    .stTextInput>div>div>input {
        background-color: white;
    }
    
    /* Sophisticated header styles */
    .header-modern {
        background-color: white;
        padding: 2rem 3rem;
        margin: -4rem -4rem 2rem -4rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .header-split {
        display: flex;
        align-items: center;
        gap: 2rem;
        padding: 1rem;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    }
    
    .header-minimal {
        padding: 2rem;
        border-bottom: 2px solid #eaeaea;
        margin-bottom: 2rem;
    }
    
    .header-image {
        max-width: 100%;
        height: auto;
        object-fit: contain;
    }
    
    .title-modern {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 2.5rem;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
        color: #1a1a1a;
    }
    
    .subtitle-modern {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        font-size: 1.1rem;
        color: #666666;
        letter-spacing: 0.01em;
        line-height: 1.5;
    }
    
    .animated-border {
        position: relative;
        overflow: hidden;
    }
    
    .animated-border::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, #0066cc, #00cc99);
        transform: translateX(-100%);
        animation: border-slide 2s ease-in-out infinite;
    }
    
    @keyframes border-slide {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    </style>
    """, unsafe_allow_html=True)




def display_tempo_title():
    """
    Display a stylish title for the TEMPO application
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
    """Render an enhanced home page with modern design elements"""
    
    # Custom CSS for enhanced styling
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

    # Header Section with Hero Image
    st.markdown("""
        <div style="text-align: center; padding: 40px 0;">
            <h1 class="gradient-text" style="font-size: 3.5em;">TEMPO</h1>
            <p style="font-size: 1.5em; color: #666;">Tracking and Estimating Motion for Patient Observation</p>
        </div>
    """, unsafe_allow_html=True)

    # Quick Stats Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="stat-box">
                <h3>50Hz</h3>
                <p>Sampling Rate</p>
            </div>
        """, unsafe_allow_html=True)


    with col2:
        st.markdown("""
            <div class="stat-box">
                <h3>85%</h3>
                <p>Motion Accuracy</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="stat-box">
                <h3>6</h3>
                <p>Motion Sensors</p>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
            <div class="stat-box">
                <h3>17</h3>
                <p>Tracked Points</p>
            </div>
        """, unsafe_allow_html=True)

    # Welcome Message
    st.markdown("""
        <div class="highlight-container">
            <h2>Welcome to TEMPO</h2>
            <p style="font-size: 1.1em; line-height: 1.6;">
                TEMPO revolutionises motion tracking in medical applications by combining 
                acessible wearable technology with advanced 3D pose estimation. Whether you're 
                a medical professional, researcher, or healthcare provider, our platform offers 
                comprehensive tools for accurate motion analysis and patient monitoring.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Main Features Section
    st.markdown("## 🚀 Key Features", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <h3>📊 Motion Tracking</h3>
                <ul>
                    <li>High-precision sensor data collection</li>
                    <li>Multi-device synchronisation</li>
                    <li>Real-time data visualisation</li>
                    <li>Customizable sampling rates</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-card" style="margin-top: 20px;">
                <h3>🔍 Data Analysis</h3>
                <ul>
                    <li>Advanced signal processing</li>
                    <li>Statistical analysis tools</li>
                    <li>Customizable reports</li>
                    <li>Export capabilities</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <h3>🎯 3D Pose Estimation</h3>
                <ul>
                    <li>Real-time pose tracking</li>
                    <li>High accuracy reconstruction</li>
                    <li>Multiple viewing angles</li>
                    <li>Motion path analysis</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-card" style="margin-top: 20px;">
                <h3>📱 Device Integration</h3>
                <ul>
                    <li>Wrist sensors support</li>
                    <li>Smartphone integration</li>
                    <li>eSense earbuds compatibility</li>
                    <li>Easy device management</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    # Quick Start Guide
    st.markdown("## 🚀 Quick Start Guide")
    
    tabs = st.tabs(["1. Setup Devices", "2. Record Data", "3. Analyze Results", "4. Generate Reports"])
    
    with tabs[0]:
        st.markdown("""
            ### Setting Up Your Devices
            - Connect your wearable sensors
            - Configure sampling rates
            - Verify connections
            - Calibrate devices
        """)
    
    with tabs[1]:
        st.markdown("""
            ### Recording Motion Data
            - Position sensors correctly
            - Start synchronized recording
            - Monitor data quality
            - Save recorded sessions
        """)
    
    with tabs[2]:
        st.markdown("""
            ### Analyzing Your Results
            - Process raw data
            - Generate 3D visualizations
            - Apply analysis algorithms
            - Review motion patterns
        """)
    
    with tabs[3]:
        st.markdown("""
            ### Generating Reports
            - Create detailed summaries
            - Export visualizations
            - Share results securely
            - Track progress over time
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
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🎯 Start Data Collection", use_container_width=True):
            st.info("Please select 'Data Analysis' from the sidebar and go to the 'Collect Device Data' tab.")
    

    
    # Add some example visualizations or statistics for the homepage
    st.subheader("System Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("📊 Data Processing Pipeline")
        st.markdown("""
        1. Collect wearable device data
        2. Process motion patterns
        3. Generate 3D pose estimation
        4. Analyze results
        """)
    
    with col2:
        st.info("🎯 Benefits")
        st.markdown("""
        - Accurate motion tracking
        - Real-time feedback
        - Comprehensive analysis
        - Patient progress monitoring
        """)


def main():
    """Main function to run the Streamlit app"""
    # Initialize session state for page navigation if it doesn't exist
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'main'
    
    # Create sidebar navigation
    st.sidebar.title("Navigation")
    
    # Check if we should redirect to data collection
    query_params = st.experimental_get_query_params()
    if query_params.get("page", [""])[0] == "data_collection":
        page = "Data Analysis"
        # Clear the query parameter - use the older method
        query_params.pop("page", None)
        st.experimental_set_query_params(**query_params)
    else:
        page = st.sidebar.radio(
            "Go to",
            ["Home", "Data Analysis", "About Us", "Documentation & Help"],
            key="nav_radio"
        )
    
    # Check if we need to override the page based on session state
    if st.session_state.current_page == 'esens_collection':
        # Import and display the esens collection page
        from frontend.pages.esens_collection import esens_collection_page
        esens_collection_page()
    else:
        # Display selected page from the sidebar
        if page == "Data Analysis":
            data_analysis_page()
        elif page == "Home":
            home_page()
        elif page == "About Us":
            about_us_page()
        else:
            documentation_help_page()
            
if __name__ == "__main__":
    main()

