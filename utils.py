import numpy as np
from nltk.tokenize import TweetTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tknzr = TweetTokenizer()
max_len = 10

def load_vec_file(file_path, limit_dims=100):
    embeddings_index = {}
    word_dict = []
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()  # Đọc dòng đầu tiên (số từ và số chiều)
        vocab_size, vector_size = map(int, first_line.split())
        print(f"Số từ: {vocab_size}, Kích thước vector: {vector_size} (Giới hạn đọc: {limit_dims} chiều)")

        for line in f:
            values = line.split()
            word = values[0]  # Từ
            # Chỉ đọc tối đa 'limit_dims' chiều
            coefs = np.asarray(values[1:1 + limit_dims], dtype='float32')
            embeddings_index[word] = coefs
            word_dict.append(word)

    return embeddings_index, word_dict


def custom_tokenizer(text_data):
    text_data = text_data.lower()
    return tknzr.tokenize(text_data)


def encode(X, word_to_index):
    sentences = []

    for t in X:
        sentences.append(custom_tokenizer(t))

    X = []
    for s in sentences:
        sent = []
        for w in s:
            try:
                w = w.lower()
                sent.append(word_to_index[w])
            except:
                sent.append(word_to_index["UNK"])
        X.append(sent)

    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=word_to_index["PAD"])

    return (X)


def decoding(text_data, encoding_text, prediction, idx2word):
    test = [[idx2word[i] for i in row] for row in encoding_text]

    lst_token = []

    for t in range(0, len(test)):
        yy_pred = []
        for i in range(0, len(test[t])):
            if prediction[t][i] == 1:
                yy_pred.append(test[t][i])
        lst_token.append(yy_pred)

    lis_idx = []
    for i in range(0, len(text_data)):
        idx = []
        for t in lst_token[i]:
            index = text_data[i].find(t)
            idx.append(index)
            for j in range(1, len(t)):
                index = index + 1
                idx.append(index)
        lis_idx.append(idx)

    return lis_idx

def tokenize_text(text):
  return tknzr.tokenize(text)

def toxic_word(span, text):
  i = 0
  token = []
  a = 0
  word = []

  while (i < (len(span) - 1)):
      if (span[i] != (span[i+1]-1)):
          token.append(span[a:(i+1)])
          a = i + 1
      elif i == (len(span) - 2):
          token.append(span[a:i+2])
      i = i + 1
  for t in token:
      word.append(text[t[0]:(t[len(t)-1])+1])
  return word


def span_convert(text_data, spans):
    MAX_LEN = 0
    token_labels = []

    for i in range(0, len(text_data)):
        token_labels.append(toxic_word(spans[i], text_data[i]))

    lst_seq = []
    for i in range(0, len(text_data)):
        token = tokenize_text(text_data[i])
        if len(token) > MAX_LEN:
            MAX_LEN = len(token)

        seq = np.zeros(len(token), dtype=int)
        for j in range(0, len(token)):
            for t in token_labels[i]:
                if token[j] in tokenize_text(t):
                    seq[j] = 1
        lst_seq.append(seq)

    return (token_labels, lst_seq)

