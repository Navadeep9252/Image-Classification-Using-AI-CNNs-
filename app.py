import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Needed for flash messages
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load pre-trained model
print("Loading ResNet50 model...")
try:
    model = ResNet50(weights='imagenet')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if model is None:
        flash('Model not loaded. Please check your setup.')
        return redirect(url_for('home'))
    
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load and preprocess the image
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Make prediction
            predictions = model.predict(img_array)
            decoded_predictions = decode_predictions(predictions, top=3)[0]
            
            # Format predictions
            results = []
            for _, label, probability in decoded_predictions:
                results.append({
                    'label': label.replace('_', ' ').title(),
                    'probability': f"{probability * 100:.2f}%"
                })
            
            # Create visualization
            img_base64 = create_visualization(img, results)
            
            return render_template('results.html', 
                                  filename=filename, 
                                  predictions=results,
                                  img_data=img_base64)
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('home'))
    else:
        flash('Allowed file types are: png, jpg, jpeg, gif')
        return redirect(request.url)

def create_visualization(original_img, predictions):
    """Create a visualization of the image with predictions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display the image
    ax1.imshow(original_img)
    ax1.axis('off')
    ax1.set_title('Uploaded Image')
    
    # Create a bar chart of predictions
    labels = [pred['label'] for pred in predictions]
    probabilities = [float(pred['probability'].strip('%')) for pred in predictions]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    bars = ax2.barh(labels, probabilities, color=colors)
    ax2.set_xlabel('Probability (%)')
    ax2.set_title('Classification Results')
    ax2.invert_yaxis()  # Display highest probability at the top
    ax2.set_xlim(0, 100)
    
    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{prob:.2f}%', ha='left', va='center')
    
    plt.tight_layout()
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    
    # Encode the image to base64 for HTML display
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64

if __name__ == '__main__':
    app.run(debug=True)