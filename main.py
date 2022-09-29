import pickle
from flask import Flask, render_template, request

# OOPS

# Create an object of the class Flask

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():

    try:
        area = request.form.get('area')
        bedrooms = request.form.get('bedrooms')
        bathrooms = request.form.get('bathrooms')
        stories = request.form.get('stories')
        air = request.form.get('airconditioning')
        parking = request.form.get('parking')
        pref = request.form.get('prefarea')
        furnishing = request.form.get('furnishingstatus')

        # print(area, bedrooms, bathrooms, stories, air, parking, pref, furnishing)

        prediction = model.predict([[area, bedrooms, bathrooms, stories, air, parking, pref, furnishing]])
        output = round(prediction[0], 2)
        return render_template('index.html', prediction_text=f'Price of the house is {output}/-')

    except:
        return render_template('index.html', prediction_text='Something went wrong!')


if __name__ == '__main__':
    app.run(debug=True)
