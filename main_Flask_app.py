from compute_lstm import TenDimensionsLSTM
from flask import Flask, request, redirect , jsonify ,send_from_directory
from flask_cors import CORS
import json


# Load model
models_dir = 'lstm_trained_models'
embeddings_dir = 'embeddings'  # change urls to embeddings dir
model = TenDimensionsLSTM(models_dir=models_dir, embeddings_dir=embeddings_dir)
print('Model loaded')

app = Flask(__name__)



@app.route("/tenDimensions", methods=['POST'])
def tenDimensions():
        sentences = [request.form['text']]
        print(f"received Text : {sentences}")
        try:
                # you can give in input both texts or a list of texts
                scores = model.compute_score(sentences, dimensions=None)
                # dimensions = None extracts all dimensions
                return jsonify(scores)
        except Exception as e:
                print (e)
                print("Something went wrong while opening summary")
                return '400'
                

if __name__ == "__main__":
        CORS(app)
        app.run(host="0.0.0.0",port=5000,threaded=True)
