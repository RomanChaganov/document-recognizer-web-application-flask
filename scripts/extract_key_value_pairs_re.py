import json
from transformers import AutoTokenizer
from nltk.tokenize import WordPunctTokenizer
from scripts.extract_text import extract_text

import os
from typing import List, Tuple, Dict

import torch

DEBUG = False
BERT_NAME = "ai-forever/ruBert-base"

entities_positions = None
tags_pos = None


class Tag:
    def __init__(self, name, pos):
        self.name = name
        self.pos = pos

    def __repr__(self):
        return f'Tag(name={self.name}, pos={self.pos})'


def get_tags_with_positions(labels, id2label):
    tags_pos = []
    ind = 0
    while ind < len(labels):
        if id2label[labels[ind]].startswith("B"):
            tag = id2label[labels[ind]].split("-")[1]
            start_pos = ind
            ind += 1
            while ind < len(labels) and id2label[labels[ind]].startswith("I"):
                ind += 1
            end_pos = ind
            tags_pos.append({"tag": tag, "pos": [start_pos, end_pos]})
        else:
            ind += 1
    return tags_pos


def get_mean_vector_from_segment(embeddings, start_pos, end_pos):
    return embeddings[start_pos:end_pos].mean(dim=0)


def tokenize(text, tokenizer, nltk_tokenizer=WordPunctTokenizer(), max_length=512):
    tokenized_text_spans = list(nltk_tokenizer.span_tokenize(text))
    words = [text[span[0]: span[1]] for span in tokenized_text_spans]
    encoded = tokenizer(words, is_split_into_words=True, add_special_tokens=False, max_length=max_length,
                        truncation=True, padding='max_length')
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
            if relation_matrix_pred[i][j] == 0:  # to do no_relation_tag
                tag1 = Tag(tags_pos[i]['tag'], entities_positions[i])
                tag2 = Tag(tags_pos[j]['tag'], entities_positions[j])
                relations.append((tag1, tag2))

    return relation_matrix_pred, relations


def extract_key_value_pairs(image, model_ner, model_re, label2id, entity_tag_to_id) -> List[Tuple[str, str]]:
    text = extract_text(image)

    print(text)

    result = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
    nltk_tokenizer = WordPunctTokenizer()

    input_ids, words_ids_for_tokens, words = tokenize(text, tokenizer, nltk_tokenizer)

    id2label = {id: label for label, id in label2id.items()}

    model_ner = model_ner.to(device)
    model_ner.eval()

    entities_description = ner_out(input_ids, model_ner, id2label, entity_tag_to_id, device)

    model_re = model_re.to(device)
    model_re.eval()

    relation_matrix_pred, relations = re_out(entities_description, model_re, device)

    i = 0
    for tag1, tag2 in relations:
        if tag1.name != 'KEY':
            continue

        start_pos1, end_pos1 = tag1.pos
        start_pos2, end_pos2 = tag2.pos

        words1 = set(words_ids_for_tokens[start_pos1:end_pos1])
        words2 = set(words_ids_for_tokens[start_pos2:end_pos2])

        # print(f"{tag1.name}: {' '.join([words[i] for i in words1])}")
        # print(f"{tag2.name}: {' '.join([words[i] for i in words2])}")

        key = ' '.join([words[i] for i in words1])
        value = ' '.join([words[i] for i in words2])
        result.append((key, value))

        i += 1
    return result

if __name__ == '__main__':
    with open('file29.txt', 'r', encoding='UTF-8') as file:
        text = file.read()
        extract_key_value_pairs(text)