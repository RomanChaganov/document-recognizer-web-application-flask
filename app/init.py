import json
import torch

from transformers import AutoTokenizer
from app.models import BertCrf, ReBertCrf


BERT_NAME = "ai-forever/ruBert-base"
BERT_CRF_PATH = "weights/bert-crf.pt"
RE_BERT = "weights/re.pt"
LABEL2ID = "data/label2id.json"

NUM_LABELS = 5
DROPOUT = 0
USE_CRF = True

NUM_RE_TAGS = 2
HIDDEN_SIZE = 768

device = "cuda" if torch.cuda.is_available() else "cpu"

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

re_model = ReBertCrf(NUM_RE_TAGS, HIDDEN_SIZE, DROPOUT, entity_tag_to_id)
re_model.load_from(RE_BERT)
re_model = re_model.to(device)
re_model.eval()

tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)  

