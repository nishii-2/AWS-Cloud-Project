
from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the processed data and the KNN model
processed_data = pd.read_csv('Processed_data.csv')
model = joblib.load('knn_model.pkl')

@app.route('/recommend', methods=['GET'])
def recommend():
    # Get the query parameters
    greV = request.args.get('greV', type=float)
    greQ = request.args.get('greQ', type=float)
    greA = request.args.get('greA', type=float)
    cgpa = request.args.get('cgpa', type=float)

    # Prepare the input data for prediction
    input_data = [[greV, greQ, greA, cgpa]]
    input_df = pd.DataFrame(input_data, columns=['greV', 'greQ', 'greA', 'cgpa'])

    # Make predictions using the trained model
    predictions = model.predict(input_df)

    # Return the results as a JSON response
    return jsonify({"recommended_university": predictions[0]})

if __name__ == '__main__':
    app.run(debug=True)
