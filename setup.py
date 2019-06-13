"""Download and pre-process SQuAD and GloVe.

Usage:
    > source activate squad
    > python setup.py

Pre-processing code adapted from:
    > https://github.com/HKUST-KnowComp/R-Net/blob/master/prepro.py

Author:
    Chris Chute (chute@stanford.edu)
"""

import numpy as np
import os
import spacy
import ujson as json
import urllib.request
import torch

from args import get_setup_args
from codecs import open
from collections import Counter
from subprocess import run
from tqdm import tqdm
from zipfile import ZipFile
from pytorch_pretrained_bert import BertTokenizer, BasicTokenizer, BertModel


def download_url(url, output_path, show_progress=True):
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if show_progress:
        # Download with a progress bar
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url,
                                       filename=output_path,
                                       reporthook=t.update_to)
    else:
        # Simple download with no progress bar
        urllib.request.urlretrieve(url, output_path)


def url_to_data_path(url):
    return os.path.join('./data/', url.split('/')[-1])


def download(args):
    downloads = [
        # Can add other downloads here (e.g., other word vectors)
        ('GloVe word vectors', args.glove_url),
    ]

    for name, url in downloads:
        output_path = url_to_data_path(url)
        if not os.path.exists(output_path):
            print('Downloading {}...'.format(name))
            download_url(url, output_path)

        if os.path.exists(output_path) and output_path.endswith('.zip'):
            extracted_path = output_path.replace('.zip', '')
            if not os.path.exists(extracted_path):
                print('Unzipping {}...'.format(name))
                with ZipFile(output_path, 'r') as zip_fh:
                    zip_fh.extractall(extracted_path)

    print('Downloading spacy language model...')
    run(['python', '-m', 'spacy', 'download', 'en'])


def word_tokenize(sent, Bert=True):
    if Bert:
        return tokenizer.tokenize(sent)
    else:
        doc = nlp(sent)
        return [token.text for token in doc]


def all_suffixes(s):
    return [s[-i:] for i in range(len(s), 0, -1)]


def find_index(text, token, current_idx):
    for s in all_suffixes(token):
        idx = text.find(s, current_idx)
        if idx >= 0:
            return idx

    return -1

def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False

def _tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def clean_text(text):
    cleaner = BasicTokenizer()
    ret_text = cleaner._run_strip_accents(text)
    ret_text = _tokenize_chinese_chars(ret_text)

    ret_text = ret_text.replace("[UNK]", " ")
    ret_text = ret_text.replace("[SEP]", " ")
    ret_text = ret_text.replace("[PAD]", " ")
    ret_text = ret_text.replace("[CLS]", " ")
    ret_text = ret_text.replace("[MASK]", " ")

    if ret_text.startswith("##"):
        ret_text = ret_text[2:]

    return ret_text.lower()


def _is_bad(token):
    return token in ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]


def convert_idx(text, tokens):
    current = 0
    spans = []
    text = clean_text(text)
    for token in tokens:
        if _is_bad(token):
            spans.append((current, current + len(token)))
            current += 1
            continue
            
        token = clean_text(token)
        current = find_index(text, token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            print("DEBUG:\n {0}\n {1}\n".format(current, text))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filename, data_type, word_counter, char_counter):
    print("Pre-processing {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                context_tokens = word_tokenize(context)
                context_chars = [list(token) for token in context_tokens]
                spans = convert_idx(context, context_tokens)
                for token in context_tokens:
                    word_counter[token] += len(para["qas"])
                    global_word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                    ques_tokens = word_tokenize(ques)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        global_word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    y1s, y2s = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = []
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                    example = {"context_tokens": context_tokens,
                               "context_chars": context_chars,
                               "ques_tokens": ques_tokens,
                               "ques_chars": ques_chars,
                               "y1s": y1s,
                               "y2s": y2s,
                               "id": total}
                    examples.append(example)
                    eval_examples[str(total)] = {"context": context,
                                                 "question": ques,
                                                 "spans": spans,
                                                 "answers": answer_texts,
                                                 "uuid": qa["id"]}
        print("{} questions in total".format(len(examples)))
    return examples, eval_examples


def _create_segment_ids(tokens):
    ids = [0] * len(tokens)

    bit = 0
    for i in range(len(tokens)):
        ids[i] = bit

        if tokens[i] == "[SEP]":
            bit = 1 - bit

    return ids


def get_bert_embedding(train_examples, dev_examples):
    print("Bert Embedding Prep-processing...")

    vec_size = 768
    embedding_dict = {}
    examples = {**train_examples, **dev_examples}

    for i in tqdm(range(len(examples))):
        example = examples[i]
        context_tokens = example["context_tokens"]
        context_ids = _create_segment_ids(context_tokens)
        ques_tokens = example["ques_tokens"]
        ques_ids = _create_segment_ids(ques_tokens)

        tokens = context_tokens + ques_tokens
        ids = context_ids + ques_ids

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([ids])

        tokens_tensor = tokens_tensor.to('cuda')
        segments_tensors = segments_tensors.to('cuda')

        with torch.no_grad():
            encoded_layers, _ = bert(tokens_tensor, segments_tensors)

            layer_8 = encoded_layers[8].tolist()[0]
            layer_9 = encoded_layers[9].tolist()[0]
            layer_10 = encoded_layers[10].tolist()[0]
            layer_11 = encoded_layers[11].tolist()[0]

            layers = [layer_8, layer_9, layer_10, layer_11]

            for layer in layers:
                for i in range(layer): # iterate over tokens
                    token = tokens[i]
                    embedding = layer[i]

                    denom = global_word_counter[token]

                    if denom == 0:
                        denom = 1
                    elif denom > 1:
                        embedding = [x / float(denom) for x in embedding]

                    embedding_dict[token] += embedding

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def get_embedding(counter, data_type, limit=-1, emb_file=None, vec_size=None, num_vectors=None):
    print("Pre-processing {} vectors...".format(data_type))
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=num_vectors):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector #KH: This is the line we should change
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.1) for _ in range(vec_size)]
        print("{} tokens have corresponding {} embedding vector".format(
            len(filtered_elements), data_type))

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def convert_to_features(args, data, word2idx_dict, char2idx_dict, is_test):
    example = {}
    context, question = data
    context = context.replace("''", '" ').replace("``", '" ')
    question = question.replace("''", '" ').replace("``", '" ')
    example['context_tokens'] = word_tokenize(context)
    example['ques_tokens'] = word_tokenize(question)
    example['context_chars'] = [list(token) for token in example['context_tokens']]
    example['ques_chars'] = [list(token) for token in example['ques_tokens']]

    para_limit = args.test_para_limit if is_test else args.para_limit
    ques_limit = args.test_ques_limit if is_test else args.ques_limit
    char_limit = args.char_limit

    def filter_func(example):
        return len(example["context_tokens"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit

    if filter_func(example):
        raise ValueError("Context/Questions lengths are over the limit")

    context_idxs = np.zeros([para_limit], dtype=np.int32)
    context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
    ques_idxs = np.zeros([ques_limit], dtype=np.int32)
    ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)

    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1

    def _get_char(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return 1

    for i, token in enumerate(example["context_tokens"]):
        context_idxs[i] = _get_word(token)

    for i, token in enumerate(example["ques_tokens"]):
        ques_idxs[i] = _get_word(token)

    for i, token in enumerate(example["context_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            context_char_idxs[i, j] = _get_char(char)

    for i, token in enumerate(example["ques_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            ques_char_idxs[i, j] = _get_char(char)

    return context_idxs, context_char_idxs, ques_idxs, ques_char_idxs


def is_answerable(example):
    return len(example['y2s']) > 0 and len(example['y1s']) > 0


def build_features(args, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):
    para_limit = args.test_para_limit if is_test else args.para_limit
    ques_limit = args.test_ques_limit if is_test else args.ques_limit
    ans_limit = args.ans_limit
    char_limit = args.char_limit

    def drop_example(ex, is_test_=False):
        if is_test_:
            drop = False
        else:
            drop = len(ex["context_tokens"]) > para_limit or \
                   len(ex["ques_tokens"]) > ques_limit or \
                   (is_answerable(ex) and
                    ex["y2s"][0] - ex["y1s"][0] > ans_limit)

        return drop

    print("Converting {} examples to indices...".format(data_type))
    total = 0
    total_ = 0
    meta = {}
    context_idxs = []
    context_char_idxs = []
    ques_idxs = []
    ques_char_idxs = []
    y1s = []
    y2s = []
    ids = []
    for n, example in tqdm(enumerate(examples)):
        total_ += 1

        if drop_example(example, is_test):
            continue

        total += 1

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        context_idx = np.zeros([para_limit], dtype=np.int32)
        context_char_idx = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idx = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idx = np.zeros([ques_limit, char_limit], dtype=np.int32)

        for i, token in enumerate(example["context_tokens"]):
            context_idx[i] = _get_word(token)
        context_idxs.append(context_idx)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idx[i] = _get_word(token)
        ques_idxs.append(ques_idx)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idx[i, j] = _get_char(char)
        context_char_idxs.append(context_char_idx)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idx[i, j] = _get_char(char)
        ques_char_idxs.append(ques_char_idx)

        if is_answerable(example):
            start, end = example["y1s"][-1], example["y2s"][-1]
        else:
            start, end = -1, -1

        y1s.append(start)
        y2s.append(end)
        ids.append(example["id"])

    np.savez(out_file,
             context_idxs=np.array(context_idxs),
             context_char_idxs=np.array(context_char_idxs),
             ques_idxs=np.array(ques_idxs),
             ques_char_idxs=np.array(ques_char_idxs),
             y1s=np.array(y1s),
             y2s=np.array(y2s),
             ids=np.array(ids))
    print("Built {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    return meta


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def pre_process(args):
    # Process training set and use it to decide on the word/character vocabularies
    word_counter, char_counter = Counter(), Counter()
    train_examples, train_eval = process_file(args.train_file, "train", word_counter, char_counter)

    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, 'char', emb_file=None, vec_size=args.char_dim)

    # Process dev and test sets
    dev_examples, dev_eval = process_file(args.dev_file, "dev", word_counter, char_counter)

    # Retrieving embedding for both Dev & Training Datasets
    word_emb_mat, word2idx_dict = get_bert_embedding(train_examples, dev_examples)

    ## Building Features for both Dev & Training Datasets
    build_features(args, train_examples, "train", args.train_record_file, word2idx_dict, char2idx_dict)
    dev_meta = build_features(args, dev_examples, "dev", args.dev_record_file, word2idx_dict, char2idx_dict)

    # Working on Test Examples
    if args.include_test_examples:
        test_examples, test_eval = process_file(args.test_file, "test", word_counter, char_counter)
        save(args.test_eval_file, test_eval, message="test eval")
        test_meta = build_features(args, test_examples, "test",
                                   args.test_record_file, word2idx_dict, char2idx_dict, is_test=True)
        save(args.test_meta_file, test_meta, message="test meta")

    save(args.word_emb_file, word_emb_mat, message="word embedding")
    save(args.char_emb_file, char_emb_mat, message="char embedding")
    save(args.train_eval_file, train_eval, message="train eval")
    save(args.dev_eval_file, dev_eval, message="dev eval")
    save(args.word2idx_file, word2idx_dict, message="word dictionary")
    save(args.char2idx_file, char2idx_dict, message="char dictionary")
    save(args.dev_meta_file, dev_meta, message="dev meta")


def LoadPretrainedBERT():
    print("Loading Pretrained Bert Model")
    bert = BertModel.from_pretrained('bert-base-uncased')

    print("Evaluating Bert Model")
    bert.eval()
    
    print("Moving Bert Model to GPU")
    bert.to('cuda')

    print("Bert Loading Completed..\n")

    return bert


if __name__ == '__main__':
    # Get command-line args
    args_ = get_setup_args()

    # Download resources
    download(args_)

    # Import spacy language model
    nlp = spacy.blank("en")

    # Load Tokenizer
    print("Loading Bert Tokenizer..")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("Done Loading Bert Tokenizer")

    # Load Bert Pretrained Model
    bert = LoadPretrainedBERT()

    global_word_counter = Counter()

    # Preprocess dataset
    args_.train_file = url_to_data_path(args_.train_url)
    args_.dev_file = url_to_data_path(args_.dev_url)
    if args_.include_test_examples:
        args_.test_file = url_to_data_path(args_.test_url)
    glove_dir = url_to_data_path(args_.glove_url.replace('.zip', ''))
    glove_ext = '.txt' if glove_dir.endswith('d') else '.{}d.txt'.format(args_.glove_dim)
    args_.glove_file = os.path.join(glove_dir, os.path.basename(glove_dir) + glove_ext)
    pre_process(args_)
