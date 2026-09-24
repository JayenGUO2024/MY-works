from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Sentiment Analysis!"})

@app.route('/data', methods=['GET'])
def get_data():
    sample_data = {
        "id": 1,
        "name": "Sample Data",
        "description": "This is a simple data object."
    }
    return jsonify(sample_data)

@app.route('/data', methods=['POST'])
def create_data():
    new_data = request.json
    return jsonify({"message": "Data created successfully!", "data": new_data}), 201

@app.route('/data/<int:data_id>', methods=['PUT'])
def update_data(data_id):
    updated_data = request.json
    return jsonify({"message": f"Data with ID {data_id} updated successfully!", "data": updated_data})

@app.route('/data/<int:data_id>', methods=['DELETE'])
def delete_data(data_id):
    return jsonify({"message": f"Data with ID {data_id} deleted successfully!"})

if __name__ == '__main__':
    app.run(debug=True)
