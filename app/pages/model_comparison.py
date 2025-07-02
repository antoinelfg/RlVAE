"""
Model Comparison Page
===================

Interface for comparing different VAE models and their performance.
"""

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, List, Any, Optional

# Add src to path
current_dir = Path(__file__).parent.parent.parent.absolute()
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def render():
    """Render the model comparison page."""
    
    st.title("📊 Model Comparison")
    st.markdown("Compare different VAE models and analyze their performance")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Load Models", 
        "📈 Performance Metrics", 
        "🌌 Latent Analysis", 
        "📋 Summary Report"
    ])
    
    with tab1:
        render_model_loading_interface()
    
    with tab2:
        render_performance_comparison()
    
    with tab3:
        render_latent_space_comparison()
    
    with tab4:
        render_comparison_report()


def render_model_loading_interface():
    """Render interface for loading models for comparison."""
    
    st.header("🎯 Load Models for Comparison")
    
    # Current loaded models
    loaded_models = st.session_state.get('loaded_models', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📂 Add New Model")
        
        # Model upload
        uploaded_file = st.file_uploader(
            "Upload Model Checkpoint",
            type=['pth', 'pt', 'ckpt'],
            help="Upload a model checkpoint for comparison"
        )
        
        model_name = st.text_input(
            "Model Name",
            value="",
            placeholder="Enter a name for this model",
            help="Unique name for this model in comparisons"
        )
        
        if uploaded_file and model_name:
            if st.button("📥 Load Model", type="primary"):
                load_model_for_comparison(uploaded_file, model_name)
        
        # Add current model
        if st.session_state.current_model is not None:
            current_name = st.text_input(
                "Add Current Model As",
                value="current_model",
                help="Name for the currently loaded model"
            )
            
            if st.button("➕ Add Current Model"):
                add_current_model_to_comparison(current_name)
    
    with col2:
        st.subheader("📋 Loaded Models")
        
        if not loaded_models:
            st.info("No models loaded for comparison yet")
        else:
            for name, model_info in loaded_models.items():
                with st.expander(f"🎯 {name}", expanded=False):
                    display_model_info(model_info)
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button(f"🔍 Analyze", key=f"analyze_{name}"):
                            analyze_single_model(name)
                    with col_b:
                        if st.button(f"🗑️ Remove", key=f"remove_{name}"):
                            remove_model_from_comparison(name)
    
    # Comparison actions
    if len(loaded_models) >= 2:
        st.subheader("🔍 Comparison Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📈 Compare Performance", type="primary"):
                st.session_state.comparison_tab = "performance"
                st.rerun()
        
        with col2:
            if st.button("🌌 Compare Latent Spaces"):
                st.session_state.comparison_tab = "latent"
                st.rerun()
        
        with col3:
            if st.button("📋 Generate Report"):
                st.session_state.comparison_tab = "report"
                st.rerun()
    else:
        st.info("Load at least 2 models to enable comparisons")


def render_performance_comparison():
    """Render performance metrics comparison."""
    
    st.header("📈 Performance Metrics Comparison")
    
    loaded_models = st.session_state.get('loaded_models', {})
    
    if len(loaded_models) < 2:
        st.warning("⚠️ Please load at least 2 models in the **Load Models** tab")
        return
    
    # Model selection for comparison
    selected_models = st.multiselect(
        "Select Models to Compare",
        options=list(loaded_models.keys()),
        default=list(loaded_models.keys()),
        help="Choose which models to include in the comparison"
    )
    
    if len(selected_models) < 2:
        st.warning("Please select at least 2 models")
        return
    
    # Metrics to compare
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Available Metrics")
        
        # Generate synthetic metrics for demonstration
        metrics_data = generate_model_metrics(selected_models)
        
        metric_types = [
            "Reconstruction Loss",
            "KL Divergence", 
            "ELBO",
            "Perplexity",
            "FID Score",
            "LPIPS Distance"
        ]
        
        selected_metrics = st.multiselect(
            "Choose Metrics",
            options=metric_types,
            default=metric_types[:3],
            help="Select metrics to compare"
        )
    
    with col2:
        st.subheader("⚙️ Comparison Settings")
        
        comparison_type = st.radio(
            "Comparison Type",
            options=["Bar Chart", "Radar Chart", "Table"],
            help="How to display the comparison"
        )
        
        normalize_metrics = st.checkbox(
            "Normalize Metrics",
            value=False,
            help="Normalize metrics to [0,1] range for better comparison"
        )
    
    # Generate comparison visualization
    if selected_metrics:
        display_performance_comparison(
            selected_models, 
            selected_metrics, 
            metrics_data, 
            comparison_type, 
            normalize_metrics
        )


def render_latent_space_comparison():
    """Render latent space analysis comparison."""
    
    st.header("🌌 Latent Space Analysis Comparison")
    
    loaded_models = st.session_state.get('loaded_models', {})
    
    if len(loaded_models) < 2:
        st.warning("⚠️ Please load at least 2 models in the **Load Models** tab")
        return
    
    # Model selection
    selected_models = st.multiselect(
        "Select Models for Latent Analysis",
        options=list(loaded_models.keys()),
        default=list(loaded_models.keys())[:2],  # Default to first 2
        help="Choose models for latent space comparison"
    )
    
    if len(selected_models) < 2:
        st.warning("Please select at least 2 models")
        return
    
    # Analysis options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎲 Sample Generation")
        
        num_samples = st.slider(
            "Number of Samples",
            min_value=50,
            max_value=500,
            value=100,
            help="Number of samples for analysis"
        )
        
        sampling_method = st.selectbox(
            "Sampling Method",
            options=["random_normal", "random_uniform", "grid"],
            help="Method for generating latent samples"
        )
    
    with col2:
        st.subheader("📊 Analysis Type")
        
        analysis_types = st.multiselect(
            "Analysis Types",
            options=[
                "Latent Distribution",
                "Reconstruction Quality", 
                "Interpolation Smoothness",
                "Latent Traversal"
            ],
            default=["Latent Distribution", "Reconstruction Quality"],
            help="Types of analysis to perform"
        )
    
    # Run analysis
    if st.button("🔍 Run Latent Space Analysis", type="primary"):
        run_latent_space_analysis(selected_models, num_samples, sampling_method, analysis_types)
    
    # Display results
    if 'latent_analysis_results' in st.session_state:
        display_latent_analysis_results()


def render_comparison_report():
    """Render comprehensive comparison report."""
    
    st.header("📋 Model Comparison Report")
    
    loaded_models = st.session_state.get('loaded_models', {})
    
    if len(loaded_models) < 2:
        st.warning("⚠️ Please load at least 2 models to generate a report")
        return
    
    # Report configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Report Settings")
        
        include_sections = st.multiselect(
            "Include Sections",
            options=[
                "Model Architecture",
                "Performance Metrics",
                "Latent Space Analysis", 
                "Reconstruction Examples",
                "Recommendations"
            ],
            default=[
                "Model Architecture", 
                "Performance Metrics", 
                "Recommendations"
            ],
            help="Choose sections to include in the report"
        )
    
    with col2:
        st.subheader("🎯 Export Options")
        
        report_format = st.selectbox(
            "Report Format",
            options=["Streamlit Display", "PDF", "HTML", "Markdown"],
            help="Format for the report"
        )
        
        if st.button("📋 Generate Report", type="primary"):
            generate_comparison_report(include_sections, report_format)
    
    # Display generated report
    if 'comparison_report' in st.session_state:
        display_comparison_report()


def load_model_for_comparison(uploaded_file, model_name: str):
    """Load a model file for comparison."""
    
    try:
        # Save uploaded file temporarily
        temp_path = Path(f"temp_{model_name}.pth")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Load model
        checkpoint = torch.load(temp_path, map_location='cpu')
        
        if 'model' in checkpoint:
            model = checkpoint['model']
        else:
            st.error("Unrecognized checkpoint format")
            temp_path.unlink()
            return
        
        # Extract model info
        model_info = extract_model_info(model, model_name)
        
        # Store in loaded models
        if 'loaded_models' not in st.session_state:
            st.session_state.loaded_models = {}
        
        st.session_state.loaded_models[model_name] = model_info
        
        # Clean up
        temp_path.unlink()
        
        st.success(f"✅ Model '{model_name}' loaded successfully!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")


def add_current_model_to_comparison(model_name: str):
    """Add the currently loaded model to comparison."""
    
    if st.session_state.current_model is None:
        st.error("No current model loaded")
        return
    
    try:
        model = st.session_state.current_model
        model_info = extract_model_info(model, model_name)
        
        if 'loaded_models' not in st.session_state:
            st.session_state.loaded_models = {}
        
        st.session_state.loaded_models[model_name] = model_info
        
        st.success(f"✅ Current model added as '{model_name}'!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Failed to add current model: {str(e)}")


def extract_model_info(model, model_name: str) -> Dict[str, Any]:
    """Extract comprehensive information from a model."""
    
    info = {
        'name': model_name,
        'model': model,
        'type': type(model).__name__
    }
    
    # Try to get model summary if available
    if hasattr(model, 'get_model_summary'):
        try:
            summary = model.get_model_summary()
            info.update(summary)
        except:
            pass
    
    # Basic attributes
    if hasattr(model, 'latent_dim'):
        info['latent_dim'] = model.latent_dim
    
    if hasattr(model, 'input_dim'):
        info['input_dim'] = model.input_dim
    
    # Count parameters
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        info['total_parameters'] = total_params
        info['trainable_parameters'] = trainable_params
    except:
        pass
    
    return info


def display_model_info(model_info: Dict[str, Any]):
    """Display model information in an expander."""
    
    st.markdown(f"**Type:** {model_info.get('type', 'Unknown')}")
    
    if 'latent_dim' in model_info:
        st.markdown(f"**Latent Dim:** {model_info['latent_dim']}")
    
    if 'input_dim' in model_info:
        st.markdown(f"**Input Shape:** {model_info['input_dim']}")
    
    if 'total_parameters' in model_info:
        st.markdown(f"**Parameters:** {model_info['total_parameters']:,}")
    
    if 'architecture' in model_info:
        arch = model_info['architecture']
        for key, value in arch.items():
            st.markdown(f"**{key}:** {value}")


def generate_model_metrics(model_names: List[str]) -> Dict[str, Dict[str, float]]:
    """Generate synthetic metrics for demonstration."""
    
    # In a real implementation, this would compute actual metrics
    metrics = {}
    
    for i, name in enumerate(model_names):
        # Generate synthetic metrics with some variation
        base_recon = 1.2 + i * 0.1 + np.random.normal(0, 0.05)
        base_kl = 0.8 + i * 0.05 + np.random.normal(0, 0.02)
        
        metrics[name] = {
            'Reconstruction Loss': max(0.1, base_recon),
            'KL Divergence': max(0.01, base_kl),
            'ELBO': -(base_recon + base_kl),
            'Perplexity': np.exp(base_kl),
            'FID Score': 15.0 + i * 2.0 + np.random.normal(0, 1.0),
            'LPIPS Distance': 0.3 + i * 0.02 + np.random.normal(0, 0.01)
        }
    
    return metrics


def display_performance_comparison(
    models: List[str], 
    metrics: List[str], 
    data: Dict[str, Dict[str, float]], 
    chart_type: str,
    normalize: bool
):
    """Display performance comparison visualization."""
    
    # Prepare data
    comparison_data = []
    for model in models:
        for metric in metrics:
            value = data[model][metric]
            comparison_data.append({
                'Model': model,
                'Metric': metric,
                'Value': value
            })
    
    df = pd.DataFrame(comparison_data)
    
    # Normalize if requested
    if normalize:
        for metric in metrics:
            metric_values = df[df['Metric'] == metric]['Value']
            min_val, max_val = metric_values.min(), metric_values.max()
            if max_val > min_val:
                df.loc[df['Metric'] == metric, 'Value'] = (metric_values - min_val) / (max_val - min_val)
    
    if chart_type == "Bar Chart":
        fig = px.bar(
            df, 
            x='Metric', 
            y='Value', 
            color='Model',
            barmode='group',
            title="Model Performance Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Radar Chart":
        # Create radar chart
        fig = go.Figure()
        
        for model in models:
            model_data = df[df['Model'] == model]
            values = model_data['Value'].tolist()
            values.append(values[0])  # Close the polygon
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, df['Value'].max() * 1.1]
                )
            ),
            title="Model Performance Radar Chart"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Table":
        # Create pivot table
        pivot_df = df.pivot(index='Metric', columns='Model', values='Value')
        
        # Style the dataframe
        styled_df = pivot_df.style.format("{:.4f}").background_gradient(axis=1)
        
        st.dataframe(styled_df, use_container_width=True)
    
    # Summary statistics
    st.subheader("📊 Summary Statistics")
    
    summary_stats = df.groupby('Model')['Value'].agg(['mean', 'std', 'min', 'max']).round(4)
    st.dataframe(summary_stats, use_container_width=True)


def run_latent_space_analysis(
    models: List[str], 
    num_samples: int, 
    sampling_method: str, 
    analysis_types: List[str]
):
    """Run latent space analysis on selected models."""
    
    try:
        loaded_models = st.session_state.loaded_models
        results = {}
        
        for model_name in models:
            model_info = loaded_models[model_name]
            model = model_info['model']
            
            # Generate samples
            latent_dim = model_info.get('latent_dim', 16)
            
            if sampling_method == "random_normal":
                samples = torch.randn(num_samples, latent_dim)
            elif sampling_method == "random_uniform":
                samples = torch.rand(num_samples, latent_dim) * 4 - 2
            elif sampling_method == "grid":
                # Create a grid of samples
                if latent_dim >= 2:
                    grid_size = int(np.sqrt(num_samples))
                    x = np.linspace(-2, 2, grid_size)
                    y = np.linspace(-2, 2, grid_size)
                    xx, yy = np.meshgrid(x, y)
                    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
                    
                    # Pad with zeros for higher dimensions
                    if latent_dim > 2:
                        padding = np.zeros((len(grid_points), latent_dim - 2))
                        grid_points = np.column_stack([grid_points, padding])
                    
                    samples = torch.FloatTensor(grid_points[:num_samples])
                else:
                    samples = torch.randn(num_samples, latent_dim)
            
            # Run analyses
            model_results = {}
            
            for analysis_type in analysis_types:
                if analysis_type == "Latent Distribution":
                    # Analyze latent distribution properties
                    model_results['distribution'] = analyze_latent_distribution(samples)
                
                elif analysis_type == "Reconstruction Quality":
                    # Analyze reconstruction quality
                    model_results['reconstruction'] = analyze_reconstruction_quality(model, samples)
                
                elif analysis_type == "Interpolation Smoothness":
                    # Analyze interpolation smoothness
                    model_results['interpolation'] = analyze_interpolation_smoothness(model, samples)
            
            results[model_name] = model_results
        
        st.session_state.latent_analysis_results = results
        st.success("✅ Latent space analysis completed!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")


def analyze_latent_distribution(samples: torch.Tensor) -> Dict[str, float]:
    """Analyze properties of latent distribution."""
    
    samples_np = samples.detach().cpu().numpy()
    
    return {
        'mean_norm': np.mean(np.linalg.norm(samples_np, axis=1)),
        'std_norm': np.std(np.linalg.norm(samples_np, axis=1)),
        'mean_activation': np.mean(samples_np),
        'std_activation': np.std(samples_np),
        'sparsity': np.mean(np.abs(samples_np) < 0.1)
    }


def analyze_reconstruction_quality(model, samples: torch.Tensor) -> Dict[str, float]:
    """Analyze reconstruction quality metrics."""
    
    try:
        with torch.no_grad():
            # Decode samples
            if hasattr(model, 'decode'):
                reconstructions = model.decode(samples)
            elif hasattr(model, 'decoder'):
                decoder_output = model.decoder(samples)
                reconstructions = decoder_output["reconstruction"] if isinstance(decoder_output, dict) else decoder_output
            else:
                return {'error': 'No decode method found'}
        
        # Simple quality metrics
        recon_np = reconstructions.detach().cpu().numpy()
        
        return {
            'mean_pixel_value': np.mean(recon_np),
            'std_pixel_value': np.std(recon_np),
            'min_pixel_value': np.min(recon_np),
            'max_pixel_value': np.max(recon_np),
            'dynamic_range': np.max(recon_np) - np.min(recon_np)
        }
        
    except Exception as e:
        return {'error': str(e)}


def analyze_interpolation_smoothness(model, samples: torch.Tensor) -> Dict[str, float]:
    """Analyze smoothness of interpolations."""
    
    try:
        # Take first two samples for interpolation
        start, end = samples[0], samples[1]
        
        # Create interpolation
        steps = 10
        alphas = torch.linspace(0, 1, steps)
        interpolated = torch.stack([
            (1 - alpha) * start + alpha * end for alpha in alphas
        ])
        
        with torch.no_grad():
            # Decode interpolated points
            if hasattr(model, 'decode'):
                decoded = model.decode(interpolated)
            elif hasattr(model, 'decoder'):
                decoder_output = model.decoder(interpolated)
                decoded = decoder_output["reconstruction"] if isinstance(decoder_output, dict) else decoder_output
            else:
                return {'error': 'No decode method found'}
        
        # Calculate smoothness metrics
        decoded_np = decoded.detach().cpu().numpy()
        
        # Compute frame-to-frame differences
        diffs = np.diff(decoded_np, axis=0)
        smoothness = np.mean(np.linalg.norm(diffs.reshape(len(diffs), -1), axis=1))
        
        return {
            'smoothness_score': smoothness,
            'max_frame_diff': np.max(np.linalg.norm(diffs.reshape(len(diffs), -1), axis=1)),
            'min_frame_diff': np.min(np.linalg.norm(diffs.reshape(len(diffs), -1), axis=1))
        }
        
    except Exception as e:
        return {'error': str(e)}


def display_latent_analysis_results():
    """Display latent space analysis results."""
    
    results = st.session_state.latent_analysis_results
    
    st.subheader("🌌 Latent Space Analysis Results")
    
    # Create comparison charts for each analysis type
    for analysis_type in ['distribution', 'reconstruction', 'interpolation']:
        if any(analysis_type in model_results for model_results in results.values()):
            render_analysis_comparison(results, analysis_type)


def render_analysis_comparison(results: Dict, analysis_type: str):
    """Render comparison for a specific analysis type."""
    
    st.subheader(f"📊 {analysis_type.replace('_', ' ').title()} Comparison")
    
    # Collect data for comparison
    comparison_data = []
    
    for model_name, model_results in results.items():
        if analysis_type in model_results and 'error' not in model_results[analysis_type]:
            for metric, value in model_results[analysis_type].items():
                comparison_data.append({
                    'Model': model_name,
                    'Metric': metric,
                    'Value': value
                })
    
    if not comparison_data:
        st.warning(f"No data available for {analysis_type} analysis")
        return
    
    df = pd.DataFrame(comparison_data)
    
    # Create comparison visualization
    fig = px.bar(
        df,
        x='Metric',
        y='Value', 
        color='Model',
        barmode='group',
        title=f"{analysis_type.replace('_', ' ').title()} Metrics Comparison"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show detailed table
    with st.expander(f"📋 Detailed {analysis_type.title()} Results", expanded=False):
        pivot_df = df.pivot(index='Metric', columns='Model', values='Value')
        st.dataframe(pivot_df.style.format("{:.4f}"), use_container_width=True)


def remove_model_from_comparison(model_name: str):
    """Remove a model from comparison."""
    
    if 'loaded_models' in st.session_state and model_name in st.session_state.loaded_models:
        del st.session_state.loaded_models[model_name]
        st.success(f"Model '{model_name}' removed from comparison")
        st.rerun()


def generate_comparison_report(sections: List[str], format_type: str):
    """Generate comprehensive comparison report."""
    
    loaded_models = st.session_state.loaded_models
    
    report = {
        'title': 'VAE Model Comparison Report',
        'models': list(loaded_models.keys()),
        'sections': {}
    }
    
    # Generate each section
    for section in sections:
        if section == "Model Architecture":
            report['sections']['architecture'] = generate_architecture_section(loaded_models)
        elif section == "Performance Metrics":
            report['sections']['performance'] = generate_performance_section(loaded_models)
        elif section == "Latent Space Analysis":
            report['sections']['latent'] = generate_latent_section()
        elif section == "Recommendations":
            report['sections']['recommendations'] = generate_recommendations_section(loaded_models)
    
    st.session_state.comparison_report = report
    st.success("✅ Comparison report generated!")


def generate_architecture_section(loaded_models: Dict) -> Dict:
    """Generate architecture comparison section."""
    
    arch_data = {}
    for name, info in loaded_models.items():
        arch_data[name] = {
            'type': info.get('type', 'Unknown'),
            'latent_dim': info.get('latent_dim', 'Unknown'),
            'parameters': info.get('total_parameters', 'Unknown'),
            'input_shape': info.get('input_dim', 'Unknown')
        }
    
    return arch_data


def generate_performance_section(loaded_models: Dict) -> Dict:
    """Generate performance comparison section."""
    
    # Use synthetic metrics for demonstration
    metrics = generate_model_metrics(list(loaded_models.keys()))
    return metrics


def generate_latent_section() -> Dict:
    """Generate latent space analysis section."""
    
    if 'latent_analysis_results' in st.session_state:
        return st.session_state.latent_analysis_results
    else:
        return {'note': 'No latent space analysis available. Run analysis first.'}


def generate_recommendations_section(loaded_models: Dict) -> Dict:
    """Generate recommendations based on analysis."""
    
    recommendations = {
        'summary': f"Compared {len(loaded_models)} VAE models",
        'best_overall': "Model comparison would determine best model here",
        'use_cases': {
            'high_quality_reconstruction': "Model with lowest reconstruction loss",
            'latent_space_structure': "Model with best latent organization", 
            'computational_efficiency': "Model with fewest parameters"
        },
        'suggestions': [
            "Consider model ensemble for improved performance",
            "Investigate hyperparameter sensitivity",
            "Evaluate on additional datasets for robustness"
        ]
    }
    
    return recommendations


def display_comparison_report():
    """Display the generated comparison report."""
    
    report = st.session_state.comparison_report
    
    st.markdown(f"# {report['title']}")
    st.markdown(f"**Models Compared:** {', '.join(report['models'])}")
    st.markdown("---")
    
    for section_name, section_data in report['sections'].items():
        if section_name == 'architecture':
            display_architecture_section(section_data)
        elif section_name == 'performance':
            display_performance_section(section_data)
        elif section_name == 'latent':
            display_latent_section(section_data)
        elif section_name == 'recommendations':
            display_recommendations_section(section_data)


def display_architecture_section(data: Dict):
    """Display architecture section of report."""
    
    st.subheader("🏗️ Model Architecture Comparison")
    
    df = pd.DataFrame(data).T
    st.dataframe(df, use_container_width=True)


def display_performance_section(data: Dict):
    """Display performance section of report."""
    
    st.subheader("📈 Performance Metrics")
    
    df = pd.DataFrame(data).T
    st.dataframe(df.style.format("{:.4f}"), use_container_width=True)


def display_latent_section(data: Dict):
    """Display latent analysis section of report."""
    
    st.subheader("🌌 Latent Space Analysis")
    
    if 'note' in data:
        st.info(data['note'])
    else:
        st.write(data)


def display_recommendations_section(data: Dict):
    """Display recommendations section of report."""
    
    st.subheader("💡 Recommendations")
    
    st.markdown(f"**Summary:** {data['summary']}")
    st.markdown(f"**Best Overall:** {data['best_overall']}")
    
    st.markdown("**Recommended Use Cases:**")
    for use_case, recommendation in data['use_cases'].items():
        st.markdown(f"- **{use_case.replace('_', ' ').title()}:** {recommendation}")
    
    st.markdown("**Additional Suggestions:**")
    for suggestion in data['suggestions']:
        st.markdown(f"- {suggestion}")


def analyze_single_model(model_name: str):
    """Analyze a single model in detail."""
    
    st.info(f"Detailed analysis for {model_name} would be implemented here")
    # This would show detailed analysis for a single model