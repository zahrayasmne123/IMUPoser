import streamlit as st # type: ignore

################ ABOUT US PAGE #######################
# 1. Title Section: Displays the TEMPO title
# 2. Highlighted container explaining the project's purpose and research overview
# 3. Two-Column Layout: Showing TEMPO's solution and info about researcher
# 4. Technical Implementation Section discussing data collection methods, processing pipeline etc
#
def about_us_page():
    """Render the About Us page with research context"""
    
    # Add custom CSS including the tempo-title styles
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
                
        .highlight-container {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #0066cc;
            margin: 10px 0;
        }
        
        .tempo-subtitle {
            font-family: 'Arial', sans-serif;
            font-size: 1.5em;
            color: #34495e;
            text-align: center;
            margin-bottom: 30px;
            font-style: italic;
        }

        .section-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        .tech-section {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin: 10px 0;
        }

        .highlight-text {
            color: #2c3e50;
            font-weight: bold;
        }

        h2 {
            color: #2c3e50;
            margin-top: 30px;
            margin-bottom: 20px;
            font-size: 1.8em;
        }

        h3 {
            color: #34495e;
            margin-top: 20px;
            font-size: 1.3em;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title Section
    st.markdown("""
        <div class="tempo-title">About TEMPO</div>
        <div class="tempo-subtitle">Tracking and Evaluating Movement for Patient Observation</div>
    """, unsafe_allow_html=True)
    
    # Introductory message
    st.markdown("""
        <div class="highlight-container">
            <h2>Research Overview</h2>
            <p style="font-size: 1.1em; line-height: 1.6;">
                TEMPO is a research project addressing the growing challenges of neurodegenerative diseases in our aging global population. 
                As part of a Computer Science dissertation at the University of Warwick, this project focuses on creating accessible solutions for long-term movement monitoring.
            Despite decades of research, neurodegenerative conditions like Alzheimer's remain without a cure. These conditions significantly impact motor function, making early detection and monitoring crucial.
                Current methods for long-term movement analysis face both usability and technological limitations. TEMPO's aim is to overcome this. </p>
        </div>
    """, unsafe_allow_html=True)


    # Create two columns
    our_solution_card, researcher_info = st.columns(2)

    # Overview Section in first column
    with our_solution_card:
        st.markdown("""
        <div class="section-card">           
        <h2>💡 Our Solution</h2>
        TEMPO bridges this gap by:
        <ul> 
            <li>Utilising everyday wearable devices for data collection</li>
            <li>Implementing IMUPoser's deep learning algorithm for 3D visualisation</li>
            <li>Providing an accessible platform for continuous monitoring</li>
        </ul>
        </div>
        """ ,unsafe_allow_html=True)

    # Research Team Section in second column
    with researcher_info:
        st.markdown("""
        <div class="section-card">
            <h2>👤 Main Researcher</h2>
            <p class="highlight-text">Zahra Rahman</p>
            <ul>
                <li>Third Year Computer Science Student</li>
                <li>University of Warwick</li>
                <li>Focus: 3D Pose Estimation using Wearable Sensors</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


    # Technical Implementation Section
    st.markdown("""
        <div class="section-card">
            <h2> ⚙️ Technical Implementation</h2>
    """, unsafe_allow_html=True)
    
    tech_section, processing_pipeline = st.columns(2)
    
    with tech_section:
        st.markdown("""
            <div class="tech-section">
                <h3>Data Collection</h3>
                <ul>
                    <li>eSense Earbuds for head movement</li>
                    <li>MetaWear wrist sensors</li>
                    <li>Smartphone sensors</li>
                    <li>50Hz sampling rate across devices</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
    with processing_pipeline:
        st.markdown("""
            <div class="tech-section">
                <h3>Processing Pipeline</h3>
                <ul>
                    <li>Deep learning-based pose estimation</li>
                    <li>Data synchronisation</li>
                    <li>3D visualisation generation</li>
                    <li>Movement analysis algorithms</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)


