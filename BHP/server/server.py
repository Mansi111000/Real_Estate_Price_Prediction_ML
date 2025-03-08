from flask import Flask, request, jsonify
import util

from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    try:
        total_sqft = float(request.form.get('total_sqft', 0))
        location = request.form.get('location')
        bhk = int(request.form.get('bhk', 0))
        bath = int(request.form.get('bath', 0))

        if not location or total_sqft <= 0 or bhk <= 0 or bath <= 0:
            return jsonify({'error': 'Invalid input parameters'}), 400

        estimated_price = util.get_estimated_price(location, total_sqft, bhk, bath)
        return jsonify({'estimated_price': estimated_price})

    except ValueError:
        return jsonify({'error': 'Invalid data format'}), 400
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    util.load_saved_artifacts()
    app.run()






