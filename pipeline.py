import glob
import json
import torch
from tqdm import tqdm

import numpy as np
from collections import defaultdict
from collections import OrderedDict
from transformers import AutoTokenizer
from nltk.tokenize import WordPunctTokenizer

from models.bert_crf import BertCrf
from models.re_bert_crf import ReBertCrf
from re_utils.ner import get_tags_with_positions, get_mean_vector_from_segment



class Tag:
    def __init__(self, name, pos):
        self.name = name
        self.pos = pos
        
    def __repr__(self):
        return f'Tag(name={self.name}, pos={self.pos})'


DEBUG = True
NUM_LABELS = 5
DROPOUT = 0
USE_CRF = True

NUM_RE_TAGS = 2
HIDDEN_SIZE = 768

BERT_NAME = "ai-forever/ruBert-base"
BERT_CRF_PATH = "weights/bert-crf.pt"
RE_BERT = "weights/re.pt"
LABEL2ID = "data/label2id.json"

entities_positions = None
tags_pos = None


def tokenize(text, tokenizer, nltk_tokenizer=WordPunctTokenizer(), max_length=512):
    tokenized_text_spans = list(nltk_tokenizer.span_tokenize(text))
    # words = [text[span[0] : span[1]] for span in tokenized_text_spans]\
    words = ['Типовая', 'межотраслевая', 'форма', '№', '1-Т', 'Утверждена', 'постановлением', 'Госкомстата', 'России', 'от', '28.11.97', '№78', 'Форма', 'по', 'ОКУД', 'ТОВАРНО-ТРАНСПОРТНАЯ', 'НАКЛАДНАЯ', '№', 'серия', 'Дата', 'составления', 'Грузоотправитель', 'Обособленное', 'подразделение', 'ООО', '"Комус"', '454008,', 'Челябинская', 'o6n.,', 'г.Челябинск,', 'ул.Автодорожная,', '19-a,', 'тел.', '8-800-200-33-83', 'факс', '8-800-200-33-83', 'по', 'ОКПО', 'рр', 'рр', 'Наименование', 'оранизации', 'адрес', 'номертелефна', 'мм', 'Грузополучатель', '000', '"УЦСБ"', '620100,', 'Свердловская', 'o6n.,', 'г.Екатеринбург,', 'ул.Ткачей,', '6,', 'тел', '3433799834', 'по', 'ОКПО', 'полное', 'наименование', 'организации,', 'адрес,', 'номер', 'телефона', 'Плательщик', '000', '"УЦСБ"', 'ИНН', '6672235068', '620100,', 'СВЕРДЛОВСКАЯ', 'ОБЛАСТЬ,', 'Г.', 'ЕКАТЕРИНБУРГ,', 'УЛ.', 'ТКАЧЕЙ,', 'Д.6', 'тел', '3433799834*1100', 'факс', '3433820563', 'р/с', '40702810900000068305', 'к/с', '30101810200000000823', 'Банк', 'ГПБ', '(АО)', 'БИК', '044525823', 'по', 'ОКПО', 'полное', 'наименование', 'организации,', 'адрес,', 'банковские', 'реквизиты', 'VPP', 'ыыы', 'оо—оо—', 'до', '—ж—ж—жжж»»—»—»—»__', 'Документ', 'об', 'отгрузке', 'товаров', 'и', 'счет-фактура', '№', '34888215', 'от', '26.02.2024', 'передаются', 'ЭЛЕКТРОННО', 'ТТН', '№', '0VT/49358868/TTH', 'от', '26/02/2024', 'no', 'a/c', '№', 'OVT/847765/50553808', 'от', '26.02.2024', 'Поставщик', 'ООО', '"КОМУС"', 'ИНН', '7721793895', 'Страница', '1', 'из', '3', 'Покупатель', 'ИНН', '6672235068', 'ООО', '"УЦСБ"']
    encoded = tokenizer(words, is_split_into_words=True, add_special_tokens=False, max_length=max_length, truncation=True, padding='max_length')
    # print(tokenizer.batch_decode(encoded['input_ids']))
    input_ids = encoded["input_ids"]
    words_ids_for_tokens = encoded.word_ids()
    
    return input_ids, words_ids_for_tokens, words
    
    
def ner_out(input_ids, model, id2label, entity_tag_to_id, device):
    global entities_positions, tags_pos
    attention_mask = torch.ones(1, len(input_ids), device=device)
    input_ids = torch.tensor([input_ids], device=device)
    
    _, batched_bert_embeddings = model.get_bert_features(input_ids, attention_mask)
    bert_embeddings = batched_bert_embeddings[0]
    full_seq_embedding = get_mean_vector_from_segment(bert_embeddings, 0, len(bert_embeddings))
    labels = model.decode(input_ids, attention_mask)[0]
    
    tags_pos = get_tags_with_positions(labels, id2label)
    entities_positions = [item["pos"] for item in tags_pos]
    entities_embeddings = torch.tensor([
        get_mean_vector_from_segment(bert_embeddings, pos[0], pos[1]).tolist() for pos in entities_positions
    ], dtype=torch.float)
    
    entities_tags = torch.tensor([entity_tag_to_id[item["tag"]] for item in tags_pos], dtype=torch.long)
    
    return full_seq_embedding, entities_embeddings, entities_tags
    
    
def re_out(entities_description, model, device):
    entities_description = map(lambda x: x.unsqueeze(0).to(device), entities_description)    
    seq_embedding, entities_embeddings, entities_tags = entities_description
    
    relation_matrix_pred = model(seq_embedding, entities_embeddings, entities_tags)
    relation_matrix_pred = relation_matrix_pred[0].argmax(dim=-1)
    
    relations = []
    
    for i in range(len(relation_matrix_pred)):
        for j in range(len(relation_matrix_pred)):
            if relation_matrix_pred[i][j] == 0: # to do no_relation_tag
                tag1 = Tag(tags_pos[i]['tag'], entities_positions[i])
                tag2 = Tag(tags_pos[j]['tag'], entities_positions[j])
                relations.append((tag1, tag2))  
    
    return relation_matrix_pred, relations

    
def run(text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)  
    nltk_tokenizer = WordPunctTokenizer()
    
    input_ids, words_ids_for_tokens, words = tokenize(text, tokenizer, nltk_tokenizer)
    print(input_ids, words_ids_for_tokens, words, sep='\n\n')
    
    with open(LABEL2ID, "r") as label2id_file:
        label2id = json.load(label2id_file)
    id2label = {id: label for label, id in label2id.items()}
    
    entity_tags_set = set()
    for label, id in label2id.items():
        if label == "O":
            continue
        entity_tags_set.add(label.split("-")[1])
    entity_tag_to_id = {tag: id for id, tag in enumerate(entity_tags_set)}
    
    model = BertCrf(NUM_LABELS, BERT_NAME, DROPOUT, USE_CRF)
    model.load_from(BERT_CRF_PATH)
    model = model.to(device)
    model.eval()
    
    entities_description = ner_out(input_ids, model, id2label, entity_tag_to_id, device)
    
    re_model = ReBertCrf(NUM_RE_TAGS, HIDDEN_SIZE, DROPOUT, entity_tag_to_id)
    re_model.load_from(RE_BERT)
    re_model = re_model.to(device)
    re_model.eval()
    
    relation_matrix_pred, relations = re_out(entities_description, re_model, device)
    
    i = 0
    for tag1, tag2 in relations:
        if tag1.name != 'KEY':
            continue
            
        start_pos1, end_pos1 = tag1.pos
        start_pos2, end_pos2 = tag2.pos

        words1 = list(OrderedDict.fromkeys(words_ids_for_tokens[start_pos1:end_pos1]))
        words2 = list(OrderedDict.fromkeys(words_ids_for_tokens[start_pos2:end_pos2]))

        print(f"{tag1.name}: {' '.join([words[i] for i in words1])}")
        print(f"{tag2.name}: {' '.join([words[i] for i in words2])}")
        print()
        
        i += 1
    
    print(f'Найдено {i} связей')
    
    print("Sexassfully!!" if i!=0 else "Not Sexassfully((")


if __name__ == '__main__':
    with open('text.txt', 'r', encoding='UTF8') as f:
        text = f.read()
        
    run(text)
