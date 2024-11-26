import json

import numpy as np
from flask import Flask, render_template, request, send_file, jsonify, send_from_directory
from io import BytesIO
from PIL import Image, ImageOps
from scripts.process_image_start_point import process_image
from scripts.extract_key_value_pairs_re import extract_key_value_pairs
import sys
import shutil
import os
from scripts.modelClasses import BertCrf, ReBertCrf


sys.path.insert(0, 'scripts')

BERT_NAME = "ai-forever/ruBert-base"
NUM_LABELS = 5
DROPOUT = 0
USE_CRF = True
BERT_CRF_PATH = "weights/bert-crf.pt"
RE_BERT = "weights/re.pt"
LABEL2ID = "data/label2id.json"
NUM_RE_TAGS = 2
HIDDEN_SIZE = 768

app = Flask(__name__)

with open(LABEL2ID, "r") as label2id_file:
    label2id = json.load(label2id_file)

entity_tags_set = set()
for label, id in label2id.items():
    if label == "O":
        continue
    entity_tags_set.add(label.split("-")[1])
entity_tag_to_id = {tag: id for id, tag in enumerate(entity_tags_set)}

model_ner = BertCrf(NUM_LABELS, BERT_NAME, DROPOUT, USE_CRF)
model_ner.load_from(BERT_CRF_PATH)

model_re = ReBertCrf(NUM_RE_TAGS, HIDDEN_SIZE, DROPOUT, entity_tag_to_id)
model_re.load_from(RE_BERT)

@app.route('/')
def index():
    return render_template('index.html')
    

def generate_table(pairs):
    text_list = ['<table>', '<tr>', '<th>Ключ</th>', '<th>Значение</th>', '</tr>']
            
    accum = []
    for key, value in pairs:
        if not key:
            accum.append(value)
            continue
            
        if len(accum) != 0:
            text_list.append('<tr>')
            text = '\n'.join(accum)
            accum = []
            text_list.append(f'<td></td>')
            text_list.append(f'<td>{text}</td>')
            text_list.append('</tr>')
            
        text_list.append('<tr>')
        text_list.append(f'<td>{key}</td>')
        text_list.append(f'<td>{value}</td>')
        text_list.append('</tr>')
    
    text_list.append('</table>')
    return text_list


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    mode = request.form.get('mode')
    file = request.files.get('file')

    if not file:
        return 'Данные отсутствуют или повреждены'

    image = Image.open(BytesIO(file.read()))
    image = image.convert('RGB')
    image = ImageOps.exif_transpose(image)
    image = np.array(image)

    image_without_tables = process_image(image)

    pairs = extract_key_value_pairs(image_without_tables, model_ner, model_re, label2id, entity_tag_to_id)

    pairs = generate_table(pairs)

    with open('bin/pairs.txt', 'w', encoding='utf-8') as f:
        f.writelines(pairs)

    shutil.make_archive('excel_tables', 'zip', 'excel')

    return send_file('excel_tables.zip', as_attachment=True)
    
    
@app.route('/get_size', methods=['GET'])
def get_size():
    dir_path = 'bin/rotated_tables'
    size = len(os.listdir(dir_path))
    return jsonify({'size': size})
    
    
@app.route('/bin/<path:filename>', methods=['GET'])
def get_image(filename):
    return send_file('bin/' + filename)
    

# @app.route('/download', methods=['GET'])
# def download():
#     return send_file(BytesIO(upload.data), download_name=upload.filename, as_attachment=True)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
