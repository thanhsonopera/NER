import pandas as pd
import tensorflow as tf
import numpy as np
import json
import random
import logging
import re
import os
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input
from keras.models import Model
from transformers import DistilBertTokenizerFast 
from transformers import TFDistilBertForTokenClassification
from transformers import DistilBertConfig, TrainingArguments, Trainer
# import torch
from keras.optimizers import Adam
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# print(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
print(tf.__version__)
df_data = pd.read_json("ner.json", lines=True)
# Thừa cột cuối Nan
df_data = df_data.drop(['extras'], axis=1)
# Chuyển '\n' -> ' '
df_data['content'] = df_data['content'].str.replace("\n", " ")
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("No GPU found")

def mergeIntervals(intervals):
    sorted_by_lower_bound = sorted(intervals, key=lambda tup: tup[0])
    merged = []

    for higher in sorted_by_lower_bound:
        if not merged:
            merged.append(higher)
        else:
            lower = merged[-1]
            if higher[0] <= lower[1]:
                if lower[2] is higher[2]:
                    upper_bound = max(lower[1], higher[1])
                    merged[-1] = (lower[0], upper_bound, lower[2])
                else:
                    if lower[1] > higher[1]:
                        merged[-1] = lower
                    else:
                        merged[-1] = (lower[0], higher[1], higher[2])
            else:
                merged.append(higher)
    return merged

def get_entities(df):
    
    entities = []
    
    for i in range(len(df)):
        entity = []
    
        for annot in df['annotation'][i]:
            try:
                ent = annot['label'][0]
                start = annot['points'][0]['start']
                end = annot['points'][0]['end'] + 1
                entity.append((start, end, ent))
            except:
                pass
    
        entity = mergeIntervals(entity)
        entities.append(entity)
    
    return entities

# df_data['entities'] = get_entities(df_data)

def convert_dataturks_to_spacy(dataturks_JSON_FilePath):
    try:
        training_data = []
        lines=[]
        with open(dataturks_JSON_FilePath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content'].replace("\n", " ")
            entities = []
            data_annotations = data['annotation']
            if data_annotations is not None:
                for annotation in data_annotations:
                    #only a single point in text annotation.
                    point = annotation['points'][0]
                    labels = annotation['label']
                    # handle both list of labels or a single label.
                    if not isinstance(labels, list):
                        labels = [labels]

                    for label in labels:
                        point_start = point['start']
                        point_end = point['end']
                        point_text = point['text']
                        
                        lstrip_diff = len(point_text) - len(point_text.lstrip())
                        rstrip_diff = len(point_text) - len(point_text.rstrip())
                        if lstrip_diff != 0:
                            point_start = point_start + lstrip_diff
                        if rstrip_diff != 0:
                            point_end = point_end - rstrip_diff
                        entities.append((point_start, point_end + 1 , label))
            training_data.append((text, {"entities" : entities}))
        return training_data
    except Exception as e:
        logging.exception("Unable to process " + dataturks_JSON_FilePath + "\n" + "error = " + str(e))
        return None

def trim_entity_spans(data: list) -> list:
    """Removes leading and trailing white spaces from entity spans.

    Args:
        data (list): The data to be cleaned in spaCy JSON format.

    Returns:
        list: The cleaned data.
    """
    invalid_span_tokens = re.compile(r'\s')

    cleaned_data = []
    for text, annotations in data:
        entities = annotations['entities']
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            valid_end = end
            while valid_start < len(text) and invalid_span_tokens.match(
                    text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(
                    text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {'entities': valid_entities}])
    return cleaned_data  

data = trim_entity_spans(convert_dataturks_to_spacy("ner.json"))


def clean_dataset(data):
    cleanedDF = pd.DataFrame(columns=["setences_cleaned"])
    sum1 = 0
    for i in tqdm(range(len(data))):
        start = 0
        emptyList = ["Empty"] * len(data[i][0].split())
#         print("Word: ", len(data[i][0].split()))
        
        numberOfWords = 0
        
        lenOfString = len(data[i][0])
#         print("Length: ", lenOfString)
        
        strData = data[i][0]
        strDictData = data[i][1]
        lastIndexOfSpace = strData.rfind(' ')
        for i in range(lenOfString):
            h = 1
            if (strData[i]==" " and strData[i+1]!=" "):
                for k,v in strDictData.items():
#                     if h == 1:
#                         print(k, ' ', v[0])
#                     h += 1
                    for j in range(len(v)):
                        entList = v[len(v)-j-1]
                        if (start>=int(entList[0]) and i<=int(entList[1])):
                            emptyList[numberOfWords] = entList[2]
                            break
                        else:
                            continue
                start = i + 1  
                numberOfWords += 1
            if (i == lastIndexOfSpace):
                for j in range(len(v)):
                        entList = v[len(v)-j-1]
                        if (lastIndexOfSpace>=int(entList[0]) and lenOfString<=int(entList[1])):
                            emptyList[numberOfWords] = entList[2]
                            numberOfWords += 1
                            
#         print(emptyList, ' ', numberOfWords)
        cleanedDF.loc[len(cleanedDF)] = [emptyList]
#         cleanedDF = cleanedDF.append(pd.Series([],  index=cleanedDF.columns ), ignore_index=True )
#         sum1 = sum1 + numberOfWords
    return cleanedDF

cleanedDF = clean_dataset(data)

unique_tags = set(cleanedDF['setences_cleaned'].explode().unique())
# pd.unique(cleanedDF['setences_cleaned'])#set(tag for doc in cleanedDF['setences_cleaned'].values.tolist() for tag in doc)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

MAX_LEN = 512
labels = cleanedDF['setences_cleaned'].values.tolist()
# print(labels[0])
# padding = post : nếu chưa đủ MAX_LEN thì sẽ thêm [tag = Empty hay 3] vào cuối array
# truncating="post" : nếu thừa thì sẽ loại bỏ các ký tự ở cuối 
tags = pad_sequences([[tag2id.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2id["Empty"], padding="post",
                     dtype="long", truncating="post")

print(unique_tags)

# transformers  from  Hugging Face

pretrained_model_name = 'distilbert-base-uncased'

# Download the pre-trained tokenizer
# tokenizer = DistilBertTokenizerFast.from_pretrained(pretrained_model_name)
# tokenizer.save_pretrained('tokenizer/')
tokenizer = DistilBertTokenizerFast.from_pretrained('tokenizer/')

label_all_tokens = True
def tokenize_and_align_labels(tokenizer, examples, tags):
    tokenized_inputs = tokenizer(examples, truncation=True, is_split_into_words=False, padding='max_length', max_length=512)
    labels = []
    for i, label in enumerate(tags):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

test = tokenize_and_align_labels(tokenizer, df_data['content'].values.tolist(), tags)
train_dataset = tf.data.Dataset.from_tensor_slices((
    test['input_ids'],
    test['labels']
))
print( np.array(test['input_ids']).shape, ' ', np.array(test['labels']).shape)

print(test['input_ids'][0], ' ', test['labels'][0])
config = DistilBertConfig( 
  _name_or_path = "distilbert-base-uncased",
  activation = "gelu",
  architectures = [
    "DistilBertForMaskedLM"
  ],
  num_labels = len(unique_tags),
  attention_dropout = 0.1,
  dim = 768,
  dropout = 0.1,
  hidden_dim = 3072,
  id2label = {
    0: "LABEL_0",
    1: "LABEL_1",
    2: "LABEL_2",
    3: "LABEL_3",
    4: "LABEL_4",
    5: "LABEL_5",
    6: "LABEL_6",
    7: "LABEL_7",
    8: "LABEL_8",
    9: "LABEL_9",
    10: "LABEL_10",
    11: "LABEL_11"
  },
  initializer_range = 0.02,
  label2id = {
    "LABEL_0": 0,
    "LABEL_1": 1,
    "LABEL_10": 10,
    "LABEL_11": 11,
    "LABEL_2": 2,
    "LABEL_3": 3,
    "LABEL_4": 4,
    "LABEL_5": 5,
    "LABEL_6": 6,
    "LABEL_7": 7,
    "LABEL_8": 8,
    "LABEL_9": 9
  },
  max_position_embeddings = 512,
  model_type = "distilbert",
  n_heads = 12,
  n_layers = 6,
  pad_token_id = 0,
  qa_dropout = 0.1,
  seq_classif_dropout = 0.2,
  sinusoidal_pos_embds = False,
  tie_weights_ = True,
  transformers_version = "4.5.1",
  vocab_size = 30522
)
model = TFDistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', config=config)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# model.summary()

# optimizer = Adam(learning_rate=1e-4 )

# model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy']) 


# model.fit( train_dataset.shuffle(1000).batch(16), epochs=3)
training_args = TrainingArguments(
    output_dir="my_awesome_wnut_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()