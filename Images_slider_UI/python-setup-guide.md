# Python Image Comparison App - Setup Guide

I've provided two different Python implementations for your image comparison tool. Choose the one that best fits your needs:

## Option 1: Flask Implementation

Flask is a lightweight web framework that's perfect for serving your image comparison tool as a web application.

### Prerequisites

- Python 3.6+
- Your image folders with the proper structure

### Step 1: Set up a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install required packages

```bash
pip install flask pillow
```

### Step 3: Create the project structure

```
your-project-folder/
├── app.py
├── templates/
│   └── index.html
├── static/
│   └── Heatmap_images/  # Symlink or copy your images here
│       ├── original_images/
│       │   └── original_img_idx0.png
│       │   └── ...
│       └── Authenticity/
│           ├── base_model/
│           │   └── heatmap_img_idx0.png
│           │   └── ...
│           ├── negative_impact_pruned_model/
│           │   └── heatmap_img_idx0.png
│           │   └── ...
│           └── noisy_pruned_model/
│               └── heatmap_img_idx0.png
│               └── ...
```

### Step 4: Create folder structure

Make sure your Streamlit app can access your images. Either:

- Create a symbolic link to your image folder in the same directory as your app.py:
  ```bash
  ln -s /path/to/your/Heatmap_images Heatmap_images
  ```
- Or copy your images to the app directory:
  ```bash
  cp -r /path/to/your/Heatmap_images .
  ```

### Step 5: Run the application

```bash
streamlit run app.py
```

Your application will automatically open in your default web browser, typically at http://localhost:8501/

### Troubleshooting

#### Images not displaying
- Check that your image paths are correct in the code
- Verify that your images follow the expected naming convention
- Make sure your Streamlit app has access permissions to read the image files

#### Performance issues with large images
- If your images are very large, you might need to resize them for better performance
- You can add an image resizing step in the code:
  ```python
  from PIL import Image
  
  # Resize image for better performance
  def resize_image(img, max_size=1000):
      w, h = img.size
      if w > max_size or h > max_size:
          ratio = min(max_size/w, max_size/h)
          new_size = (int(w*ratio), int(h*ratio))
          return img.resize(new_size, Image.LANCZOS)
      return img
      
  # Then use this function before displaying:
  base_img = resize_image(Image.open(base_img_path))
  ```

## Differences Between Flask and Streamlit

### Flask:
- More traditional web application
- Better for production use
- More control over HTML/CSS/JavaScript
- Requires more code but gives more flexibility
- Better for long-term, production-ready applications

### Streamlit:
- Simpler to set up and run
- Designed specifically for data applications
- Less code required (more "Python-like")
- Great for prototyping and internal tools
- Automatic deployment options (Streamlit Cloud)
- Less flexibility but faster development

## Customizing the Application

Both implementations can be customized:

- Change image file extensions if your images aren't PNGs
- Adjust the UI colors and layout
- Add additional features like image zooming or annotations
- Add export capabilities to save comparison images Set up your image files

Either:
- Create a symbolic link to your existing image folder:
  ```bash
  ln -s /path/to/your/Heatmap_images static/Heatmap_images
  ```
- Or copy your images to the static folder:
  ```bash
  cp -r /path/to/your/Heatmap_images static/
  ```

### Step 5: Copy the code

- Copy the content of `python-flask-implementation.py` to `app.py`
- Copy the content of `python-flask-templates.html` to `templates/index.html`

### Step 6: Run the application

```bash
flask run
```

Your application will be available at http://127.0.0.1:5000/

