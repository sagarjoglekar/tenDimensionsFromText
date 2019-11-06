
from compute_lstm import TenDimensionsLSTM


# Load model
models_dir = 'lstm_trained_models'
embeddings_dir = 'embeddings'  # change urls to embeddings dir
model = TenDimensionsLSTM(models_dir=models_dir, embeddings_dir=embeddings_dir)
print('Model loaded')


sentences = ['How are you? I really hope you feel better now',
             'What have you just said? Your opinions are very silly, I do not want to listen anymore',
             'The Unix operating system was created in November 1971',
             'All our employees know what they do, they must be trusted',
             'Oh man, I laughed so hard at his joke that I spit my coffee',
             'This is a tradition typical of my people',
             'I desire you, my love',
             'Thank to all employees, they have done a fantastic job!'
            ]

# you can give in input both texts or a list of texts
scores = model.compute_score(sentences, dimensions=None) # dimensions = None extracts all dimensions
for sent, score in zip(sentences, scores):
    print(f'{sent} -- {score}')
