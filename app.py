from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer

from utils import load_vec_file, encode, decoding, span_convert

app = Flask(__name__)

model_path = './assets/model.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

file_path = "./assets/cc.vi.300.vec"
embeddings_index, word_dict = load_vec_file(file_path, limit_dims=100)
print(f"Đã tải {len(embeddings_index)} từ vào embeddings_index.")

words = word_dict
num_words = len(words)

word_to_index = {w : i + 2 for i, w in enumerate(words)}
word_to_index["UNK"] = 1
word_to_index["PAD"] = 0
idx2word = {i: w for w, i in word_to_index.items()}


@app.route('/comment/predict-toxic', methods=['POST'])
def predict():
    data = request.get_json()
    inputs = pd.Series(data['inputs'])
    inputs_encode = encode(inputs, word_to_index)
    inputs_encode = np.array(inputs_encode, dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], inputs_encode)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = np.argmax(output_data, axis=-1)
    output_decode = decoding(inputs, inputs_encode, output_data, idx2word)
    text_output = inputs.values
    token_predict, seq_predict = span_convert(text_output, output_decode)
    return jsonify(token_predict)


if __name__ == '__main__':
    app.run()
