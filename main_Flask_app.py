from tendims import TenDimensionsClassifier
from flask import Flask, request, redirect , jsonify ,send_from_directory
from flask_cors import CORS
import json


# Load models
models_dir = 'models/lstm_trained_models'
embeddings_dir = 'embeddings'  # change urls to embeddings dir
success_model_file = 'models/meeting_success/xgboost_10dims_success_prediction_model.dat'
model = TenDimensionsClassifier(models_dir=models_dir, embeddings_dir=embeddings_dir)
success_predictor = SuccessPredictor(success_model_file)
print('Models loaded')

app = Flask(__name__)

@app.route("/tenDimensions", methods=['POST'])
def tenDimensions():
        text = [request.form['text']]
        print(f"received Text : {sentences}")
        try:
                # you can give in input one string of text
                # dimensions = None extracts all dimensions
                tendim_scores = model.compute_score(text, dimensions=None)
                success_probability = success_predictor.predict_success(tendim_scores)
                tendim_scores['success'] = success_probability
                return jsonify(scores)
        except Exception as e:
                print (e)
                print("Something went wrong while opening summary")
                return '400'
                

if __name__ == "__main__":
        CORS(app)
        app.run(host="0.0.0.0",port=5000,threaded=True)
