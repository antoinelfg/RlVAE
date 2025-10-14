# Scene 1: VAE Fundamentals & Metric Extraction - Slides

## 🎯 Overview

This directory contains the slides for Scene 1 of the RlVAE presentation, covering:
- Vanilla VAE architecture fundamentals
- Metric extraction process via K-means clustering
- Riemannian geometry introduction

## 📁 Files

- **`scene1_slides.html`** - Interactive HTML presentation
- **`scenes/vae_metric_simple.py`** - Manim source code (simplified version)
- **`scenes/vae_metric_extraction.py`** - Full Manim source code (with LaTeX)

## 🚀 How to View the Slides

### Option 1: Direct HTML (Recommended)
1. Open `scene1_slides.html` in any modern web browser
2. Use navigation buttons or keyboard controls:
   - **Right Arrow** or **Spacebar**: Next slide
   - **Left Arrow**: Previous slide
   - **Navigation buttons**: Click to navigate

### Option 2: Local Server
```bash
cd manim
python -m http.server 8000
# Then open http://localhost:8000/scene1_slides.html in your browser
```

### Option 3: File Protocol
- Double-click `scene1_slides.html` to open directly in your default browser

## 🎨 Features

### Interactive Elements
- **Persistent Scheme**: Shows RlVAE pipeline overview in top-right corner
- **Dynamic Highlighting**: Current stage is highlighted based on slide content
- **Responsive Design**: Works on desktop and mobile devices
- **Smooth Transitions**: Professional slide transitions

### Content Structure
1. **Slide 1**: Title and introduction
2. **Slide 2**: VAE architecture visualization
3. **Slide 3**: Metric extraction process
4. **Slide 4**: K-means clustering overview

### Color Scheme
- **Encoder**: Blue (#2196F3)
- **Latent Space**: Purple (#9C27B0)
- **Decoder**: Orange (#FF9800)
- **Input/Output**: Yellow (#FBBF24)
- **Background**: Dark gradient (#0d1117 → #1f2937)

## 🔧 Customization

### Adding New Slides
1. Copy an existing slide div
2. Update the slide ID and content
3. Add navigation logic in JavaScript
4. Update the total slide count

### Modifying Colors
- Edit CSS variables in the `<style>` section
- Colors are defined using the RlVAE color scheme
- Maintain consistency with other scenes

### Adding Animations
- The HTML version is static for compatibility
- For animations, use the Manim version (`vae_metric_simple.py`)

## 📱 Browser Compatibility

- **Chrome/Edge**: Full support
- **Firefox**: Full support
- **Safari**: Full support
- **Mobile browsers**: Responsive design included

## 🎬 Manim Version

The Manim version (`vae_metric_simple.py`) provides:
- Animated transitions
- Mathematical expressions
- Professional video output
- Integration with full presentation

### Running Manim Version
```bash
cd manim
source /scratch/alaforgu/miniconda3/etc/profile.d/conda.sh
conda activate base
manim -pql --save_last_frame scenes/vae_metric_simple.py VAEMetricExtractionSimple
```

## 🔗 Integration

This scene integrates with:
- **Persistent Scheme**: Shows current progress in RlVAE pipeline
- **Color Scheme**: Consistent with other presentation components
- **Navigation**: Compatible with full presentation structure

## 📝 Notes

- The HTML version is optimized for easy sharing and viewing
- The Manim version provides professional animations
- Both versions maintain the same content structure
- Persistent scheme updates automatically based on slide content

## 🚀 Next Steps

After reviewing Scene 1:
1. **Scene 2**: Riemannian Geometry Concepts
2. **Scene 3**: RlVAE Architecture Overview
3. **Scene 4**: Flow Sequence Progression
4. **Scene 5**: Training Process
5. **Scene 6**: Results & Evaluation

---

**Created**: September 1, 2025  
**Version**: 1.0  
**Status**: ✅ Complete and Tested

