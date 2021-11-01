from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('LinearRegression.pkl', 'rb'))
df = pd.read_csv('Clean_car.csv')


@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(df['company'].unique())
    car_model = sorted(df['name'].unique())
    fuel_type = sorted(df['fuel_type'].unique())
    years = sorted(df['year'].unique(), reverse=True)
    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_model=car_model,
                           years=years, fuel_type=fuel_type)


@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kms_driven'))

    prediction = model.predict(pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]],
                                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))

    return str(np.round(prediction[0], 2))


if __name__ == '__main__':
    app.run(debug=True)
