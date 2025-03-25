# app.py
from flask import Flask, render_template, jsonify
import os
import re

app = Flask(__name__)

# Configuration
IMAGE_FOLDER = 'Heatmap_images'
BASE_MODEL_FOLDER = os.path.join(IMAGE_FOLDER, 'Authenticity', 'base_model')
NEGATIVE_IMPACT_FOLDER = os.path.join(IMAGE_FOLDER, 'Authenticity', 'negative_impact_pruned_model')
NOISY_PRUNED_FOLDER = os.path.join(IMAGE_FOLDER, 'Authenticity', 'noisy_pruned_model')
ORIGINAL_FOLDER = os.path.join(IMAGE_FOLDER, 'original_images')

@app.route('/')
def index():
    """Main page of the application."""
    return render_template('index.html')

@app.route('/api/image-info')
def get_image_info():
    """API endpoint to get image information and max index."""
    # Get list of files in base model directory
    try:
        files = os.listdir(BASE_MODEL_FOLDER)
        
        # Extract indices from file names
        pattern = re.compile(r'heatmap_img_idx(\d+)')
        indices = []
        for file in files:
            match = pattern.match(file)
            if match:
                indices.append(int(match.group(1)))
        
        # Get max index
        max_index = max(indices) if indices else 0
        
        return jsonify({
            'max_index': max_index,
            'indices': sorted(indices)
        })
    except Exception as e:
        print(f"Error getting image info: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
