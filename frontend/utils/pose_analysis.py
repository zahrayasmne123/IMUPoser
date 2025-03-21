import streamlit as st
import os
from frontend.analysis.joints import process_joint_angles
from frontend.analysis.speed import process_movement_speed
from frontend.analysis.accuracy import process_pose_accuracy
import torch
import numpy as np
import pandas as pd

def pose_analysis(base_dir):
    # Page header
    st.header("Pose Analysis")
    
    # Check if predictions exist
    predictions_path = os.path.join(base_dir, 'rawdata/processed/predictions.pt')
    if not os.path.exists(predictions_path):
        st.info("No predictions data found. Please upload and process data first to view analysis results.")
        return

    # Load predictions once to avoid reloading for each section
    try:
        predictions = torch.load(predictions_path)
    except Exception as e:
        st.error(f"Error loading predictions: {str(e)}")
        st.exception(e)
        return
        
    # Introduction
    st.write("Select analysis types below to explore different aspects of the motion data.")
    
    # Create a radio selector for analysis type
    analysis_type = st.radio(
        "Choose Analysis Type:",
        ["📐 Joint Angle Analysis", "⚡ Movement Speed Analysis", "✓ Pose Accuracy Metrics"],
        horizontal=True
    )
    
    st.divider()
    
    # Joint Angles Section
    if analysis_type == "📐 Joint Angle Analysis":
        try:
            st.subheader("Joint Angle Analysis")
            st.write("The joint angle analysis shows how the angles between connected body segments change over time using the dot product. For example, the knee angle is calculated between hip, knee, and ankle joints.")
            
            angles = process_joint_angles(predictions)
            
            if angles:
                # Group joints for easier viewing
                joint_groups = {
                    "Upper Body": [joint for joint in angles.keys() if any(x in joint for x in ["shoulder", "elbow", "wrist", "neck", "collar"])],
                    "Lower Body": [joint for joint in angles.keys() if any(x in joint for x in ["hip", "knee", "ankle", "foot"])],
                    "Spine": [joint for joint in angles.keys() if any(x in joint for x in ["back", "spine", "neck"])],
                    "All": list(angles.keys())
                }
                
                # Let user choose group - with unique key
                selected_group = st.selectbox(
                    "Select joint group:",
                    options=list(joint_groups.keys()),
                    index=0,
                    key="angles_group_selector"  # Unique key
                )
                
                # Get available joints in the selected group
                available_joints = joint_groups[selected_group] # type: ignore
                
                if available_joints:
                    # Create columns for average joint angles
                    st.write("### Average Joint Angles")
                    
                    # Determine number of columns (3 is usually a good fit)
                    num_cols = min(len(available_joints), 3)
                    cols = st.columns(num_cols)
                    
                    # Display metrics for each joint in the group
                    for i, joint in enumerate(available_joints):
                        angle_values = angles[joint]
                        with cols[i % num_cols]:
                            st.metric(
                                f"{joint.replace('_', ' ').title()}",
                                f"{np.mean(angle_values):.1f}°",
                                f"Range: {np.min(angle_values):.1f}° - {np.max(angle_values):.1f}°"
                            )
                    
                    # Plot selected joint angles over time
                    st.write("### Joint Angles Over Time")
                    
                    # Default selection - choose a reasonable number of joints
                    default_selection = available_joints[:min(3, len(available_joints))]
                    
                    # Unique key for multiselect
                    selected_joints = st.multiselect(
                        "Select joints to display:",
                        options=available_joints,
                        default=default_selection,
                        key="angles_joint_selector"  # Unique key
                    )
                    
                    if selected_joints:
                        angle_data = {joint: angles[joint] for joint in selected_joints}
                        angle_df = pd.DataFrame(angle_data)
                        st.line_chart(angle_df)
                        
                        # Add interpretation guide as regular text
                        st.write("### Joint Angle Interpretation Guide")
                        st.write("""
                        - **Higher angle values** generally indicate more extension or straightening of the joint
                        - **Lower angle values** generally indicate more flexion or bending of the joint
                        - **Stable angle values** indicate the joint is being held steady
                        - **Regular oscillations** in angle values often correspond to repetitive movements like walking
                        - **Comparing left and right sides** can help identify asymmetries in movement
                        """)
                else:
                    st.info(f"No joint angles available for the {selected_group} group.")
            else:
                st.info("No joint angle data available.")
                
        except Exception as e:
            st.error(f"Error processing joint angles: {str(e)}")
            st.exception(e)
    
    # Movement Speed Section
    elif analysis_type == "⚡ Movement Speed Analysis":
        try:
            st.subheader("Movement Speed Analysis")
            st.write("The speed graph shows the velocity (in centimeters per second) of joints over time throughout your motion sequence.")
            
            speeds, stats = process_movement_speed(predictions)
            
            # Create chart data
            speed_data = pd.DataFrame(speeds)
            
            # Group joints for easier viewing
            joint_groups = {
                "Upper Body": [col for col in speed_data.columns if any(x in col for x in ["shoulder", "elbow", "wrist", "hand", "neck", "collar", "head"])],
                "Lower Body": [col for col in speed_data.columns if any(x in col for x in ["hip", "knee", "ankle", "foot"])],
                "Spine": [col for col in speed_data.columns if any(x in col for x in ["pelvis", "spine"])],
                "All": speed_data.columns.tolist()
            }
            
            # Let user choose group - with unique key
            selected_group = st.selectbox(
                "Select joint group:",
                options=list(joint_groups.keys()),
                index=0,
                key="speed_group_selector"  # Unique key
            )
            
            # Get available joints in the selected group
            available_joints = joint_groups[selected_group] # type: ignore
            
            # Create columns for showing top movement metrics
            if stats:
                # Sort by average speed to show the most active joints
                sorted_stats = {k: stats[k] for k in sorted(stats.keys(), 
                                                          key=lambda x: stats[x]['average'] if x in stats else 0, 
                                                          reverse=True)}
                
                # Get top 3 active joints from the selected group
                top_joints = [j for j in sorted_stats.keys() if j in available_joints][:3]
                
                if top_joints:
                    st.write("### Most Active Joints")
                    metric_cols = st.columns(min(len(top_joints), 3))
                    
                    for i, joint in enumerate(top_joints):
                        with metric_cols[i % len(metric_cols)]:
                            st.metric(
                                f"{joint.replace('_', ' ').title()}",
                                f"{stats[joint]['average']:.2f} cm/s",
                                f"Peak: {stats[joint]['max']:.2f} cm/s"
                            )
            
            # Display chart with options - with unique key
            selected_parts = st.multiselect(
                "Select joints to display:",
                options=available_joints,
                default=available_joints[:3] if len(available_joints) >= 3 else available_joints,
                key="speed_joint_selector"  # Unique key
            )
            
            if selected_parts:
                filtered_data = speed_data[selected_parts]
                st.line_chart(filtered_data)
                
            st.write("### Interpretation Guide")
            st.write("- **Peaks in the graph:** These represent moments of fast movement for that joint")
            st.write("- **Valleys or low points:** These show when that joint is moving slowly or is relatively still")
            st.write("- **Compare different joints:** Look for patterns in how joints move together or independently")
                
        except Exception as e:
            st.error(f"Error processing movement speeds: {str(e)}")
            st.exception(e)
    
    # Pose Accuracy Section
    elif analysis_type == "✓ Pose Accuracy Metrics":
        try:
            st.subheader("Pose Accuracy Analysis")
            st.write("These metrics evaluate the overall quality of movement patterns, looking at smoothness, symmetry, and posture.")
            
            # Process pose accuracy
            metrics, stats = process_pose_accuracy(predictions)
            
            # Group metrics for easier viewing
            metric_groups = {
                "Overall Metrics": ["smoothness", "symmetry_score", "posture_score"],
                "Smoothness Metrics": ["smoothness", "upper_body_smoothness", "lower_body_smoothness", "torso_smoothness"],
                "Symmetry Metrics": ["symmetry_score", "arm_symmetry", "leg_symmetry"],
                "Posture Metrics": ["posture_score", "head_tilt", "vertical_alignment"],
                "All Metrics": list(metrics.keys())
            }
            
            # Let user choose group - with unique key
            selected_group = st.selectbox(
                "Select metric group:",
                options=list(metric_groups.keys()),
                index=0,
                key="accuracy_group_selector"  # Unique key
            )
            
            # Get available metrics in the selected group
            available_metrics = [m for m in metric_groups[selected_group] if m in metrics] # type: ignore
            
            if available_metrics:
                # Show statistics
                st.write("### Accuracy Statistics")
                
                # Create columns for each metric
                num_cols = min(len(available_metrics), 3)
                cols = st.columns(num_cols)
                
                for i, metric in enumerate(available_metrics):
                    if metric in stats:
                        metric_stats = stats[metric]
                        with cols[i % num_cols]:
                            st.metric(
                                label=metric.replace('_', ' ').title(),
                                value=f"{metric_stats['average']:.2f}",
                                delta=f"Range: {metric_stats['min']:.2f} - {metric_stats['max']:.2f}"
                            )
                
                # Show time-series data
                st.write("### Metrics Over Time")
                
                # Default selection - choose a reasonable number of metrics
                default_selection = available_metrics[:min(3, len(available_metrics))]
                
                # Unique key for multiselect
                selected_metrics = st.multiselect(
                    "Select metrics to display:",
                    options=available_metrics,
                    default=default_selection,
                    key="accuracy_metric_selector"  # Unique key
                )
                
                if selected_metrics:
                    metrics_df = pd.DataFrame({m: metrics[m] for m in selected_metrics})
                    st.line_chart(metrics_df)
                    
                    # Add detailed explanation as regular text instead of nested expander
                    st.write("### Metrics Explained")
                    explanations = {
                        "smoothness": "Measures motion smoothness, indicating control and coordination. Higher values indicate smoother movements with less jerky motions.",
                        "upper_body_smoothness": "Measures the smoothness of movements in the upper body (shoulders, arms, head). Higher values indicate smoother upper body movements.",
                        "lower_body_smoothness": "Measures the smoothness of movements in the lower body (hips, legs). Higher values indicate smoother lower body movements.",
                        "torso_smoothness": "Measures the smoothness of movements in the torso (pelvis, spine). Higher values indicate smoother torso movements.",
                        "symmetry_score": "Measures overall left-right body symmetry. Higher values indicate more symmetrical movement patterns.",
                        "arm_symmetry": "Measures symmetry between left and right arms. Higher values indicate more symmetrical arm movements.",
                        "leg_symmetry": "Measures symmetry between left and right legs. Higher values indicate more symmetrical leg movements.",
                        "posture_score": "Measures overall posture alignment, especially in the spine. Higher values indicate better posture alignment.",
                        "head_tilt": "Measures how vertically aligned the head is. Higher values indicate the head is more upright.",
                        "vertical_alignment": "Measures how vertically aligned the body is from pelvis to head. Higher values indicate better vertical alignment."
                    }
                    
                    for metric in selected_metrics:
                        if metric in explanations:
                            st.write(f"**{metric.replace('_', ' ').title()}**: {explanations[metric]}")
                        else:
                            st.write(f"**{metric.replace('_', ' ').title()}**: No detailed explanation available.")
                    
                    st.write("### Score Interpretation")
                    st.write("""
                    - All metrics are normalized to a 0-1 scale, where 1 is optimal
                    - Values above 0.8 generally indicate excellent performance
                    - Values between 0.6-0.8 indicate good performance
                    - Values between 0.4-0.6 indicate average performance
                    - Values below 0.4 may indicate areas for improvement
                    """)
            else:
                st.info(f"No metrics available for the {selected_group} group.")
                
        except Exception as e:
            st.error(f"Error processing pose accuracy: {str(e)}")
            st.exception(e)