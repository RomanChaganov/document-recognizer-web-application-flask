import os
from typing import List, Tuple, Dict

import torch
from torch import nn
from transformers import AutoModel


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


class BertCrf(nn.Module):
    def __init__(
        self,
        num_labels: int,
        bert_name: str,
        dropout: float = 0.2,
        use_crf: bool = True,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.use_crf = use_crf
        self.cross_entropy = nn.CrossEntropyLoss()

        self.bert = AutoModel.from_pretrained(bert_name)

        self.dropout = nn.Dropout(dropout)
        self.hidden2label = nn.Linear(self.bert.config.hidden_size, num_labels)

        self.start_transitions = nn.Parameter(torch.empty(num_labels))
        self.end_transitions = nn.Parameter(torch.empty(num_labels))
        self.transitions = nn.Parameter(torch.empty(num_labels, num_labels))

        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def _compute_log_denominator(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq_len = features.shape[0]

        log_score_over_all_seq = self.start_transitions + features[0]

        for i in range(1, seq_len):
            next_log_score_over_all_seq = torch.logsumexp(
                log_score_over_all_seq.unsqueeze(2) + self.transitions + features[i].unsqueeze(1),
                dim=1,
            )
            log_score_over_all_seq = torch.where(
                mask[i].unsqueeze(1),
                next_log_score_over_all_seq,
                log_score_over_all_seq,
            )
        log_score_over_all_seq += self.end_transitions
        return torch.logsumexp(log_score_over_all_seq, dim=1)

    def _compute_log_numerator(self, features: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq_len, bs, _ = features.shape

        score_over_seq = self.start_transitions[labels[0]] + features[0, torch.arange(bs), labels[0]]

        for i in range(1, seq_len):
            score_over_seq += (
                self.transitions[labels[i - 1], labels[i]] + features[i, torch.arange(bs), labels[i]]
            ) * mask[i]
        seq_lens = mask.sum(dim=0) - 1
        last_tags = labels[seq_lens.long(), torch.arange(bs)]
        score_over_seq += self.end_transitions[last_tags]
        return score_over_seq

    def get_bert_features(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.bert(input_ids, attention_mask=attention_mask)["last_hidden_state"]
        hidden = self.dropout(hidden)
        return self.hidden2label(hidden), hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        features, _ = self.get_bert_features(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.bool()

        if self.use_crf:
            features = torch.swapaxes(features, 0, 1)
            attention_mask = torch.swapaxes(attention_mask, 0, 1)
            labels = torch.swapaxes(labels, 0, 1)

            log_numerator = self._compute_log_numerator(features=features, labels=labels, mask=attention_mask)
            log_denominator = self._compute_log_denominator(features=features, mask=attention_mask)

            return torch.mean(log_denominator - log_numerator)
        else:
            return self.cross_entropy(
                features.flatten(end_dim=1),
                torch.where(attention_mask.bool(), labels, -100).flatten(end_dim=1),
            )

    def _viterbi_decode(self, features: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        seq_len, bs, _ = features.shape

        log_score_over_all_seq = self.start_transitions + features[0]

        backpointers = torch.empty_like(features)

        for i in range(1, seq_len):
            next_log_score_over_all_seq = (
                log_score_over_all_seq.unsqueeze(2) + self.transitions + features[i].unsqueeze(1)
            )

            next_log_score_over_all_seq, indices = next_log_score_over_all_seq.max(dim=1)

            log_score_over_all_seq = torch.where(
                mask[i].unsqueeze(1),
                next_log_score_over_all_seq,
                log_score_over_all_seq,
            )
            backpointers[i] = indices

        backpointers = backpointers[1:].int()

        log_score_over_all_seq += self.end_transitions
        seq_lens = mask.sum(dim=0) - 1

        best_paths = []
        for seq_ind in range(bs):
            best_label_id = torch.argmax(log_score_over_all_seq[seq_ind]).item()
            best_path = [best_label_id]

            for backpointer in reversed(backpointers[: seq_lens[seq_ind]]):
                best_path.append(backpointer[seq_ind][best_path[-1]].item())

            best_path.reverse()
            best_paths.append(best_path)

        return best_paths

    def decode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[List[int]]:
        features, _ = self.get_bert_features(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.bool()

        if self.use_crf:
            features = torch.swapaxes(features, 0, 1)
            mask = torch.swapaxes(attention_mask, 0, 1)
            return self._viterbi_decode(features=features, mask=mask)
        else:
            labels = torch.argmax(features, dim=2)
            predictions = []
            for i in range(len(labels)):
                predictions.append(labels[i][attention_mask[i]].tolist())
            return predictions

    def save_to(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_from(self, path: str, device: str = 'cpu'):
        self.load_state_dict(torch.load(path, map_location=torch.device(device)))


class ReBertCrf(nn.Module):
    def __init__(
            self,
            num_re_tags: int,
            hidden_size: int,
            dropout: float,
            entity_tag_to_id: Dict[str, int],
    ):
        super().__init__()

        self.arg1_linear = self.__get_dropour_relu_linear(hidden_size, hidden_size, dropout)
        self.arg2_linear = self.__get_dropour_relu_linear(hidden_size, hidden_size, dropout)
        self.all_seq_linear = self.__get_dropour_relu_linear(hidden_size, hidden_size, dropout)

        self.relation_classifier = nn.Sequential(
            nn.Linear(3 * hidden_size, 4 * hidden_size),
            # nn.LayerNorm(2 * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(4 * hidden_size, 4 * hidden_size),
            # nn.LayerNorm(4 * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(4 * hidden_size, 4 * hidden_size),
            # nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(4 * hidden_size, num_re_tags),
        )

        self.tag_embeddings = nn.Embedding(num_embeddings=len(entity_tag_to_id), embedding_dim=hidden_size)

    def __get_dropour_relu_linear(self, from_dim: int, to_dim: int, dropout: float):
        return nn.Sequential(nn.Dropout(dropout), nn.ReLU(), nn.Linear(from_dim, to_dim))

    def forward(
            self,
            seq_embedding: torch.Tensor,
            entities_embeddings: torch.Tensor,
            entities_tags: torch.Tensor,
    ):
        bs, seq_len, emb_size = entities_embeddings.shape

        entities_embeddings_as_arg1 = self.arg1_linear(entities_embeddings)
        entities_embeddings_as_arg2 = self.arg2_linear(entities_embeddings)
        tag_embeddings = self.tag_embeddings(entities_tags)

        entities_embeddings_as_arg1 += tag_embeddings
        entities_embeddings_as_arg2 += tag_embeddings

        seq_embedding_matrix = seq_embedding.unsqueeze(1).unsqueeze(1).repeat(1, seq_len, seq_len, 1)
        grid_arg1_ind, grid_arg2_ind = torch.meshgrid(torch.arange(seq_len), torch.arange(seq_len))
        concatenated_embeddings = torch.cat(
            (
                entities_embeddings[:, grid_arg1_ind, :],
                entities_embeddings[:, grid_arg2_ind, :],
                seq_embedding_matrix,
            ),
            dim=-1,
        )
        predictions = self.relation_classifier(concatenated_embeddings)
        return predictions

    def save_to(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_from(self, path: str, device: str = 'cpu'):
        self.load_state_dict(torch.load(path, map_location=torch.device(device)))
