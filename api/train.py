import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import TFBertForSequenceClassification as Bert
from tensorflow_addons.optimizers import AdamW

# load dataset
# features => 'text', labels => 'data_practice'
dataset = pd.read_csv('./opp115.csv', sep=',', header=0)
# TODO clean dataset['text']

# tokenize text, convert to sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dataset['text'].values)

features = tokenizer.texts_to_sequences(dataset['text'].values)
features = pad_sequences(features, maxlen=100) # TODO determine best pad position (pre/post)

labels = pd.get_dummies(dataset['data_practice']).values


# split into train/test subsets
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=69)

# create and train model
model = Bert.from_pretrained('bert-base-uncased', num_labels=10)
model.compile(optimizer=AdamW(weight_decay=0.01, lr=1e-5), loss='categorical_crossentropy')
model.fit(x_train, y_train, epochs=1, batch_size=32)
