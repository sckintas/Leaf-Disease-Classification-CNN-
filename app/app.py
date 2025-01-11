from flask import Flask, render_template, request
import os
from model import image_pre, predict  # Import functions from model.py

# Initialize the Flask app
app = Flask(__name__)

# Directory to save uploaded images
UPLOAD_FOLDER = r'C:\Users\OWNER\Desktop\Leaf\app\static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html', result=None, image_url=None)

@app.route('/', methods=['POST'])
def upload_file():
    """Handle file uploads and make predictions."""
    if 'file1' not in request.files:
        return render_template('index.html', result="No file uploaded!", image_url=None)
    file1 = request.files['file1']
    if file1.filename == '':
        return render_template('index.html', result="No file selected!", image_url=None)
    if not allowed_file(file1.filename):
        return render_template('index.html', result="File type not allowed!", image_url=None)
    
    # Save the uploaded file
    image_filename = 'uploaded_image.jpg'
    path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    file1.save(path)

    # Preprocess and predict
    image_data = image_pre(path)  # Preprocess the image
    prediction = predict(image_data)  # Predict the class
    
    # Pass the uploaded image's URL and prediction to the template
    image_url = f'static/{image_filename}'
    return render_template('index.html', result=prediction, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
