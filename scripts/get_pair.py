from difflib import get_close_matches
from scripts.kie_config import key_words


def get_pair(data_string: str):
    data_string_split = data_string.split()
    value = data_string.split()
    not_key_words_in_a_row = 0
    key = ''
    for i, word in enumerate(data_string_split):
        ans = get_close_matches(word.lower(), key_words, n=1, cutoff=0.7)
        if len(ans) != 0:
            data_string_split[i] = ans[0]
            last_key_index = i+1
            not_key_words_in_a_row = 0
            key = ' '.join(data_string_split[:last_key_index])
            value = data_string_split[last_key_index:]
        else:
            not_key_words_in_a_row += 1
        if not_key_words_in_a_row == 4:
            return key, ' '.join(value)
    return key, ' '.join(value)
