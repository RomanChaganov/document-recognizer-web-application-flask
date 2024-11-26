from collections import OrderedDict
from app import init
from app.models import get_tags_with_positions, get_mean_vector_from_segment

from PIL import ImageDraw

import torch


class Tag:
    def __init__(self, name, pos):
        self.name = name
        self.pos = pos
        
    def __repr__(self):
        return f'Tag(name={self.name}, pos={self.pos})'
        

def tokenize(words, tokenizer, max_length=512):
    #tokenized_text_spans = list(nltk_tokenizer.span_tokenize(text))
    #words = [text[span[0] : span[1]] for span in tokenized_text_spans]
    encoded = tokenizer(words, is_split_into_words=True, add_special_tokens=False, max_length=max_length, truncation=True, padding='max_length')
    input_ids = encoded["input_ids"]
    words_ids_for_tokens = encoded.word_ids()
    
    return input_ids, words_ids_for_tokens, words


def ner_re(input_ids, ner_model, re_model, id2label, entity_tag_to_id, device):
    attention_mask = torch.ones(1, len(input_ids), device=device)
    input_ids = torch.tensor([input_ids], device=device)
    
    _, batched_bert_embeddings = ner_model.get_bert_features(input_ids, attention_mask)
    bert_embeddings = batched_bert_embeddings[0]
    full_seq_embedding = get_mean_vector_from_segment(bert_embeddings, 0, len(bert_embeddings))
    labels = ner_model.decode(input_ids, attention_mask)[0]
    
    tags_pos = get_tags_with_positions(labels, id2label)
    entities_positions = [item["pos"] for item in tags_pos]
    entities_embeddings = torch.tensor([
        get_mean_vector_from_segment(bert_embeddings, pos[0], pos[1]).tolist() for pos in entities_positions
    ], dtype=torch.float)
    
    entities_tags = torch.tensor([entity_tag_to_id[item["tag"]] for item in tags_pos], dtype=torch.long)
    
    seq_embedding = full_seq_embedding.unsqueeze(0).to(device)
    entities_embeddings = entities_embeddings.unsqueeze(0).to(device)
    entities_tags = entities_tags.unsqueeze(0).to(device)
    
    relation_matrix_pred = re_model(seq_embedding, entities_embeddings, entities_tags)
    relation_matrix_pred = relation_matrix_pred[0].argmax(dim=-1)
    
    relations = []
    
    for i in range(len(relation_matrix_pred)):
        for j in range(len(relation_matrix_pred)):
            if relation_matrix_pred[i][j] == 0: # to do no_relation_tag
                tag1 = Tag(tags_pos[i]['tag'], entities_positions[i])
                tag2 = Tag(tags_pos[j]['tag'], entities_positions[j])
                relations.append((tag1, tag2))
    
    return tags_pos, relations


def relation_extract(words_tuple, image):
    text = [x[1] for x in words_tuple]
    positions = [x[2] for x in words_tuple]
    
    input_ids, words_ids_for_tokens, words = tokenize(text, init.tokenizer)
    tags_pos, relations = ner_re(
        input_ids, init.model, init.re_model, init.id2label, 
        init.entity_tag_to_id, init.device)
        
    visualize(image, tags_pos, relations, positions, words_ids_for_tokens)
    
    return get_key_value_dict(relations, words_ids_for_tokens, text)
    
    
def get_key_value_dict(relations, words_ids_for_tokens, text):
    result = list()
    for tag1, tag2 in relations:
        if tag1.name != 'KEY':
            continue
        
        words1 = get_words(tag1.pos, words_ids_for_tokens)
        words2 = get_words(tag2.pos, words_ids_for_tokens)
        
        key = ' '.join([text[i] for i in words1])
        value = ' '.join([text[i] for i in words2])
        
        result.append((key, value))
    
    return result
    

def get_contours(tags, words_ids_for_tokens):
    for tag in tags:
        name, pos = tag['tag'], tag['pos']
        words_ids = get_words(pos, words_ids_for_tokens)
        
        yield (words_ids, name)


def get_words(pos, words_ids_for_tokens):
    words_ids = words_ids_for_tokens[pos[0]:pos[1]]
    words_ids = list(OrderedDict.fromkeys(words_ids))
    return words_ids
    

def draw_rect(cord, draw, color):
    x0, y0, x1, y1 = cord
    x0, y0, x1, y1 = x0-5, y0-5, x1+5, y1+5
    draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
    

def draw_line(cord, draw, delta):
    x0, y0, x1, y1 = cord
    yt0, yt1 = y0 - delta, y1 - delta
    y = min(yt0, yt1)

    draw.line(((x0, y0), (x0, y), (x1, y), (x1, y1)), fill='green', width=3)


def visualize(image, tags_pos, relations, positions, words_ids_for_tokens):
    draw = ImageDraw.Draw(image)  
    
    for ids, name in get_contours(tags_pos, words_ids_for_tokens):
        color = 'blue' if name == 'KEY' else 'red'
        
        for ind in ids:
            draw_rect(positions[ind], draw, color)
    
    for tag1, tag2 in relations:
        words_ids1 = get_words(tag1.pos, words_ids_for_tokens)
        words_ids2 = get_words(tag2.pos, words_ids_for_tokens)
        
        x0, y0, x1, y1 = positions[words_ids1[0]]
        x0, y0, x1, y1 = x0-5, y0-5, x1+5, y1+5
        
        x0_avg, y0_t = (round((x0 + x1) / 2), y0)
        
        x0, y0, x1, y1 = positions[words_ids2[0]]
        x0, y0, x1, y1 = x0-5, y0-5, x1+5, y1+5
        
        x1_avg, y1_t = (round((x0 + x1) / 2), y0)
        
        draw_line((x0_avg, y0_t, x1_avg, y1_t), draw, 10)
