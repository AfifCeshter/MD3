#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import os
import nltk
import json
import numpy as np
import zipfile
from tqdm import tqdm
from functools import reduce
import unicodedata


class PreProcess:
    """
    Preprocessing
    """
    DOC_PATH = 'data/guessmovie_doc.json'
    ENTITY_PATH = 'data/entities.txt'
    AGENT_TEMP_PATH = 'data/agent_template.json'
    SIMULATOR_TEMP_PATH = 'data/simulator_template.json'
    GLOVE_PATH = '/home/lh/data-emb/glove.840B.300d.txt'

    EMB_DIM = 300

    def __init__(self, out_path):
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        self.doc_id_path = out_path + 'guessmovie_doc_id.json'
        self.vocab_path = out_path + 'vocab.txt'
        self.save_emb_path = out_path + 'gm_glove.840B.300d.npy'
        self.oov_path = out_path + 'oov.txt'

        with open(self.DOC_PATH, 'r') as f:
            self.doc_data = json.load(f)

        self.entity_vocab = None
        self.vocab = None

    def pre_process_entity(self, replace_ent):
        self.doc_data = self.do_lower()

        if replace_ent:
            self.entity_vocab = self.load_entities()
            self.doc_data = self.replace_doc_entity()
        self.doc_data = self.split_doc()

        # vocabulary with embeddings
        vocab_words = self.make_vocab()
        self.vocab = Vocabulary(vocab_words)

        # save vocabulary
        self.vocab.save_vocab(self.vocab_path)

        # transform documents to id
        self.doc_data = self.doc_to_id()

        # handle embeddings
        self.vocab.handle_glove(emb_dim=self.EMB_DIM,
                                glove_path=self.GLOVE_PATH,
                                save_glove_path=self.save_emb_path,
                                oov_path=self.oov_path)

    def save_entity(self):
        """
        save entity vocabulary
        :return:
        """
        with open(self.ENTITY_PATH, 'w') as wf:
            w_entity_vocab = list(map(lambda x: x + '\n', self.entity_vocab))
            wf.writelines(w_entity_vocab)

    def load_entities(self):
        """
        Load entities from text file
        :param path:
        :return:
        """
        with open(self.ENTITY_PATH, 'r') as f:
            lines = f.readlines()
            entities = [e.lower().rstrip() for e in lines]

        # sort with length
        entities = list(set(entities))
        entities.sort(key=lambda x: len(x), reverse=True)

        return entities

    def find_entity(self):
        """
        Get all the entity words of the KB
        :return:
        """
        entity_vocab = set()
        for d in self.doc_data:
            for k, v in d['kb'].items():
                entity_vocab.update(set(v))

            name = d['name'].replace('_', ' ')
            entity_vocab.add(name)

        # filter only with whitespace
        # entity_vocab = list(filter(lambda x: ' ' in x, entity_vocab))

        # sort with length
        entity_vocab = list(entity_vocab)
        entity_vocab.sort(key=lambda x: len(x), reverse=True)

        return entity_vocab

    def replace_doc_entity(self):
        """
        Replace all the entity in documents to special id
        :return:
        """
        new_entity_vocab = list(map(lambda x: x.replace(' ', '_'), self.entity_vocab))
        single_doc_data = json.dumps(self.doc_data, ensure_ascii=False)

        i = 0
        for entity in tqdm(self.entity_vocab, desc='Replacing entity to id'):
            single_doc_data = single_doc_data.replace(entity, new_entity_vocab[i])
            i += 1

        new_doc_data = json.loads(single_doc_data)

        return new_doc_data

    def split_doc(self):
        """
        Split the documents to words with whitespace
        :return:
        """
        new_doc = []
        for ele in tqdm(self.doc_data, desc='Documents tokenizing'):
            doc_sents = ele['doc'].split('\n')

            doc_words = [nltk.word_tokenize(s) for s in doc_sents]
            doc_words = reduce(lambda x, y: x + [Vocabulary.SEP] + y, doc_words)

            ele['doc'] = doc_words
            new_doc.append(ele)

        return new_doc

    def doc_to_id(self):
        """
        Transfer documents to id representation
        :return:
        """
        new_doc = []
        for ele in tqdm(self.doc_data, desc='Documents to id'):
            cur_doc_split = ele['doc']
            ele['doc_id'] = self.vocab.sentence_to_id(cur_doc_split)
            new_doc.append(ele)

        with open(self.doc_id_path, 'w') as wf:
            json.dump(new_doc, wf, indent=2)

        return new_doc

    def do_lower(self):
        """
        Lower all the words in documents
        :return:
        """
        return json.loads(json.dumps(self.doc_data, ensure_ascii=False).lower())

    def make_vocab(self):
        """
        Get the vocabulary of documents, dialogs and KBs
        :return:
        """
        all_vocab = set()

        # for documents
        for ele in tqdm(self.doc_data, desc='Handling documents data'):
            all_vocab.update(ele['doc'])

        # for handling template
        def handle_template(temp_path):
            with open(temp_path, 'r') as f:
                temp_data = json.load(f)

            for v in temp_data.values():
                nl_lst = []
                if 'nl_first' in temp_data:
                    nl_lst += v['nl_first']
                if 'nl_more' in temp_data:
                    nl_lst += v['nl_more']

                for nl in nl_lst:
                    all_vocab.update(Tokenizer.lower_tokenize(nl))

        # for agent & simulator template
        handle_template(self.AGENT_TEMP_PATH)
        handle_template(self.SIMULATOR_TEMP_PATH)

        # for dialogs
        # with open(self.dialog_path, 'r') as f:
        #     dialog_data = json.load(f)
        #
        # for ele in tqdm(dialog_data, desc='Handling dialog data'):
        #     dialog = ele['dialog']
        #     for i in range(len(dialog) - 1):
        #         turn = dialog[i]
        #         all_vocab.update(Tokenizer.lower_tokenize(turn['user_nl']))
        #         all_vocab.update(Tokenizer.lower_tokenize(turn['agent_nl']))

        return list(all_vocab)


class Tokenizer:
    """
    Tokenization for documents data
    """
    @staticmethod
    def lower_tokenize(turn):
        # turn = turn.replace('_', ' ')
        turn_lower = turn.lower()
        return nltk.word_tokenize(turn_lower)

    @staticmethod
    def _is_punctuation(char):
        """
        Whether a char is a punctuation
        :param char:
        :return:
        """
        # treat as part of words
        if char == '_':
            return False

        """Checks whether `chars` is a punctuation character."""
        cp = ord(char)
        # We treat all non-letter/number ASCII as punctuation.
        # Characters such as "^", "$", and "`" are not in the Unicode
        # Punctuation class but we treat them as punctuation anyways, for
        # consistency.
        if ((33 <= cp <= 47) or (58 <= cp <= 64) or
                (91 <= cp <= 96) or (123 <= cp <= 126)):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    @staticmethod
    def split_sentence(s):
        """
        (Not used)
        Split a sentence to words with whitespace
        :param s:
        :return:
        """
        output = ''

        for c in s:
            if Tokenizer._is_punctuation(c):
                if len(output) and output[-1] != ' ':
                    output += ' '
                output += c
                output += ' '
            elif c == ' ' and output[-1] == ' ':
                continue
            else:
                output += c

        return output


class Vocabulary:
    """
    Construct vocabulary for word and id transforming
    """
    PAD_IDX = 0
    SEP_IDX = 4
    PAD = '_PAD_'
    BOS = '_BOS_'
    EOS = '_EOS_'
    OOV = '_OOV_'
    SEP = '_SEP_'

    def __init__(self, vocab_arg):
        if type(vocab_arg) == str:
            self.vocab = self.read_vocab(vocab_arg)
        elif type(vocab_arg) == list:
            # PAD should be the first index
            self.vocab = [self.PAD, self.BOS, self.EOS, self.OOV, self.SEP, '_UNUSED1_', '_UNUSED2_']

            # make sure unique words
            for w in self.vocab:
                if w in vocab_arg:
                    vocab_arg.remove(w)
            self.vocab += vocab_arg
        else:
            raise ValueError('Not valid argument for vocab_arg with %s' % vocab_arg)

        self.word2id = dict(zip(self.vocab,
                                range(len(self.vocab))))

    def __len__(self):
        return len(self.vocab)

    def read_vocab(self, vocab_path):
        """
        read the vocab from vocabulary path
        :return:
        """
        vocab = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                vocab.append(line.strip())
        return vocab

    def save_vocab(self, vocab_path):
        self._save_words(vocab_path, self.vocab)

    def _save_words(self, path, words):
        with open(path, 'w', encoding='utf-8') as wf:
            wvocab = list(map(lambda x: x + '\n', words))
            wf.writelines(wvocab)

    # def word_to_id(self, word):
    #     if word not in self.vocab:
    #         return self.vocab.index(self.OOV)
    #
    #     return self.vocab.index(word)

    def word_to_id(self, word):
        if word not in self.vocab:
            return self.word2id[self.OOV]
        return self.word2id[word]

    def sentence_to_id(self, sentence):
        return list(map(lambda w: self.word_to_id(w), sentence))

    def handle_glove(self, emb_dim, glove_path, save_glove_path, oov_path):
        """
        handle glove embeddings: reading and saving embeddings
        :return:
        """
        # reading the embeddings
        print("read glove from text file %s" % glove_path)
        embeddings = np.random.rand(len(self), emb_dim)
        non_oov_words = []

        # with zipfile.ZipFile(glove_path, 'r') as zf:
        #     if len(zf.namelist()) != 1:
        #         raise ValueError('glove file "%s" not recognized' % glove_path)
        #
        #     glove_name = zf.namelist()[0]

        with open(glove_path, 'r') as f:
            for line in tqdm(f):
                line_split = line.split(' ')
                word = line_split[0]
                vec = [float(x) for x in line_split[1:]]

                if word in self.vocab:
                    embeddings[self.word2id[word]] = vec
                    non_oov_words.append(word)
        oov_words = list(set(self.vocab).difference(set(non_oov_words)))

        # save the embeddings
        print("save glove to file %s" % save_glove_path)
        np.save(save_glove_path, embeddings)

        print('OOV words num: %d/%d' % (len(oov_words), len(self)))
        self._save_words(oov_path, oov_words)
