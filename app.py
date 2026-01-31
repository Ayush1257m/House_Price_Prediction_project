from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
df = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl",'rb'))

@app.route('/')
def index():

    locations = sorted(df['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')
    balcony = request.form.get('balcony')
    
    if not all([location, bhk, bath, sqft, balcony]):
        return "Error: All fields are required. Please fill in all the details."
    
    try:
        bhk = int(bhk)
        bath = int(bath)
        sqft = float(sqft)
        balcony = int(balcony)

        print(location, bhk, bath, sqft, balcony)
        input = pd.DataFrame([[location,sqft,bath,bhk,balcony]], columns=['location', 'total_sqft', 'bath', 'bhk','balcony'])
        prediction = pipe.predict(input)[0] * 1e5

        return str(np.round(prediction,2))
    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True, port=5002)