from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and preprocessor
model = pickle.load(open(r'c:\Users\mnrth\Documents\Jupyter Notebooks\Pro\Crop Yeild Prediction\dtr.pkl', 'rb'))
preprocessor = pickle.load(open(r'c:\Users\mnrth\Documents\Jupyter Notebooks\Pro\Crop Yeild Prediction\preprocessor.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    year = int(request.form.get('year'))
    rainfall = float(request.form.get('rainfall'))
    pesticides = float(request.form.get('pesticides'))
    temperature = float(request.form.get('temperature'))
    area = request.form.get('area')
    crop = request.form.get('crop')
    
    # Convert input into a feature array
    input_features = np.array([[area, crop, year, rainfall, pesticides, temperature]])
    processed_features = preprocessor.transform(input_features)
    
    # Predict using the model
    prediction = model.predict(processed_features)
    predicted_value = prediction[0]
    
    return render_template('index.html', prediction=f'Predicted crop yield: {predicted_value:.2f} hectogram/ hectare')

if __name__ == '__main__':
    app.run(debug=True)
