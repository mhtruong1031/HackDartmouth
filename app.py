from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from pipeline import Pipeline
import os
import mimetypes

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config['UPLOAD_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

pl = Pipeline()

# List of allowed video MIME types
ALLOWED_EXTENSIONS = {
    'video/mp4',
    'video/mpeg',
    'video/quicktime',
    'video/x-msvideo',
    'video/x-ms-wmv',
    'video/webm',
    'video/x-matroska'
}

def allowed_file(file):
    # Check the file's MIME type
    mime_type = file.content_type
    
    # Check if it's a video file
    if mime_type in ALLOWED_EXTENSIONS:
        return True
    
    # Double-check using the file extension as backup
    filename = file.filename.lower()
    valid_extensions = ('.mp4', '.mpeg', '.mov', '.avi', '.wmv', '.webm', '.mkv')
    return any(filename.endswith(ext) for ext in valid_extensions)

# Ensure the static folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/app')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'video' not in request.files:
            return 'No file part'
        file = request.files['video']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            file.save('static/temp.mp4')
            response = pl.run('static/temp.mp4')

            with open('temp.txt', 'w') as f:
                f.write(response)

            return f'File uploaded successfully: {filename}'
    return render_template('upload.html')

@app.route('/static/latest_video')
def latest_video():
    try:
        files = [
            f for f in os.listdir(app.config['UPLOAD_FOLDER'])
            if f.lower().endswith(('.mp4', '.mpeg', '.mov', '.avi', '.wmv', '.webm', '.mkv'))
        ]
        if not files:
            return {'filename': None}
        
        # Sort by most recent modified time
        files.sort(key=lambda x: os.path.getmtime(os.path.join(app.config['UPLOAD_FOLDER'], x)), reverse=True)
        latest = files[0]
        print('balls')
        return {'filename': latest}
    except Exception as e:
        return {'filename': None}
    
@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/analysis')
def analysis():
    with open('temp.txt', 'r') as f:
        m = f.readline().strip()

    return render_template('Analysis.html', data = {'message':m})

@app.route('/faq')  
def faq():
    return render_template('FAQ.html')

if __name__ == '__main__':
    app.run(debug=True) 