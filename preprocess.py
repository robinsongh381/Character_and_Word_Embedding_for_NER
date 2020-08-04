import utils.constant as config
from tqdm import tqdm
import torch
import glob,re
from konlpy.tag import Mecab, Okt, Kkma
okt = Okt()

# Entitiy-index dictionary
global_entity_dict = torch.load('./data/processed_data/entity_to_index.pt')

# Load raw data 
train_set = glob.glob('./data/raw_data/train_set/*.txt')
valid_set = glob.glob('./data/raw_data/validation_set/*.txt')

# Process raw data and save indexed .pt files
reg_label = re.compile('<(.+?):[A-Z]{3}>') # detect texts with entity tag
reg_idx = re.compile('## \d+$') # detect texts without entity tag


class Vocabulary():
    def __init__(self):
        self.vocab = set()
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.add_special()
        
    def add_special(self):
        special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        for word in special_tokens:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.vocab.add(word)
            self.idx += 1
        
    def add_word(self, tokenized_text, char=False, pos_tag=False):
        for word in tokenized_text:
            
            if not char and not pos_tag:
                if word not in self.vocab:
                    self.word2idx[word] = self.idx
                    self.idx2word[self.idx] = word
                    self.vocab.add(word)
                    self.idx += 1
                    
            elif char and not pos_tag:
                for c in word:
                    if c not in self.vocab:
                        self.word2idx[c] = self.idx
                        self.idx2word[self.idx] = c
                        self.vocab.add(c)
                        self.idx += 1
                        
            elif not char and pos_tag:
                for pos in word[1]:
                    if pos not in self.vocab:
                        self.word2idx[pos] = self.idx
                        self.idx2word[self.idx] = pos
                        self.vocab.add(pos)
                        self.idx += 1
                            
    def convert_tokens_to_idx(self, list_of_tokens, add_special=False):
        list_of_idx = []
        for w in list_of_tokens:
            try:
                idx = self.word2idx[w]
            except:
                idx = self.word2idx['<unk>']
            list_of_idx.append(idx)
            
        return list_of_idx
 
    def convert_chars_to_idx(self, list_of_tokens, add_special=False):
        list_of_idx = []
        for w in list_of_tokens:
            char= []
            for c in w:
                try:
                    idx = self.word2idx[c]
                except:
                    idx = self.word2idx['<unk>']
                char.append(idx)
            list_of_idx.append(char)
            
        return list_of_idx
    
    def convert_pos_to_idx(self, raw_text, add_special=False):
        list_of_idx = []
        pos_tag = okt.pos(raw_text)

        for p in pos_tag:
            pos = p[1]
            try:
                idx = self.word2idx[pos]
            except:
                idx = self.word2idx['<unk>']
            list_of_idx.append(idx)
            
        return list_of_idx
                    
    def __len__(self):
        return len(self.word2idx)


def tokenizer(raw_text):
    return okt.morphs(raw_text)


def convert_entity_to_idx(ner_tag):
    label = []
    for tag in ner_tag:
        label.append(global_entity_dict[tag])
    return label


def transform_source_fn(raw_text):
    start_index = []
    prev = 0
    # tokens = tok(text)
    tokenized_text = tokenizer(raw_text)
    start = 0
    for i, token in enumerate(tokenized_text):
        if i == 0:
            start_index.append(0)
            start += len(token)
        else:
            whitespace = raw_text[:start+1].count(' ')
            if whitespace != prev:
                prev= whitespace
                start +=1
            start_index.append(start)
            start += len(token)
            
    return tokenized_text, start_index


def transform_target_fn(label_text, tokens, prefix_sum_of_token_start_index):

    regex_ner = re.compile('<(.+?):[A-Z]{3}>') # NER Tag가 2자리 문자면 {3} -> {2}로 변경 (e.g. LOC -> LC) 인경우
    regex_filter_res = regex_ner.finditer(label_text)

    list_of_ner_tag = []
    list_of_ner_text = []
    list_of_tuple_ner_start_end = []


    count_of_match = 0
    for match_item in regex_filter_res:
        ner_tag = match_item[0][-4:-1]  # <4일간:DUR> -> DUR
        ner_text = match_item[1]  # <4일간:DUR> -> 4일간
        start_index = match_item.start() - 6 * count_of_match  # delete previous '<, :, 3 words tag name, >'
        end_index = match_item.end() - 6 - 6 * count_of_match

        list_of_ner_tag.append(ner_tag)
        list_of_ner_text.append(ner_text)
        list_of_tuple_ner_start_end.append((start_index, end_index))
        count_of_match += 1

    list_of_ner_label = []
    entity_index = 0
    is_entity_still_B = True
    for tup in zip(tokens, prefix_sum_of_token_start_index):
        token, index = tup

        if '▁' in token:  # 주의할 점!! '▁' 이것과 우리가 쓰는 underscore '_'는 서로 다른 토큰임
            index += 1    # 토큰이 띄어쓰기를 앞단에 포함한 경우 index 한개 앞으로 당김 # ('▁13', 9) -> ('13', 10)

        if entity_index < len(list_of_tuple_ner_start_end):
            start, end = list_of_tuple_ner_start_end[entity_index]

            if end < index:  # 엔티티 범위보다 현재 seq pos가 더 크면 다음 엔티티를 꺼내서 체크
                is_entity_still_B = True
                entity_index = entity_index + 1 if entity_index + 1 < len(list_of_tuple_ner_start_end) else entity_index
                start, end = list_of_tuple_ner_start_end[entity_index]

            if start <= index and index < end:  
                entity_tag = list_of_ner_tag[entity_index]
                if is_entity_still_B is True:
                    entity_tag = 'B-' + entity_tag
                    list_of_ner_label.append(entity_tag)
                    is_entity_still_B = False
                else:
                    entity_tag = 'I-' + entity_tag
                    list_of_ner_label.append(entity_tag)
            else:
                is_entity_still_B = True
                entity_tag = 'O'
                list_of_ner_label.append(entity_tag)
        else:
            entity_tag = 'O'
            list_of_ner_label.append(entity_tag)
    
    return list_of_ner_label


# Build Vocab
word_vocab = Vocabulary()
char_vocab = Vocabulary()
pos_vocab = Vocabulary()

mode = ['train']
for m in mode:
    if m=='train':
        dataset = train_set
        print('Processing {} training data...'.format(len(dataset)))

    for file in dataset:
        with open(file, "r", encoding = "utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n','')

                if reg_idx.search(line): ## 1
                    continue

                elif line[:2]=='##' and not reg_label.search(line) : # raw_text
                    word_vocab.add_word(okt.morphs(line[3:])) # without ##
                    char_vocab.add_word(okt.morphs(line[3:]), char=True)
                    pos_vocab.add_word(okt.pos(line[3:]), pos_tag=True)
                        
    print('Word vocab size : {}'.format(len(word_vocab.word2idx)))       
    print('Char vocab size : {}'.format(len(char_vocab.word2idx)))
    print('POS vocab size : {}'.format(len(pos_vocab.word2idx))) 
    
torch.save(word_vocab.word2idx,'./data/word_vocab.pt')
torch.save(char_vocab.word2idx,'./data/char_vocab.pt')
torch.save(pos_vocab.word2idx,'./data/pos_vocab.pt')


tr_token_idx = []
tr_char_idx = []
tr_pos_idx = []
tr_ner_idx = []

valid_token_idx = []
valid_char_idx = []
valid_pos_idx = []
valid_ner_idx = []


mode = ['tr', 'valid']
for m in mode:
    if m=='tr':
        dataset = train_set
        save_token = tr_token_idx
        save_char = tr_char_idx
        save_pos = tr_pos_idx
        save_ner = tr_ner_idx
        print('Processing {} training data...'.format(len(dataset)))
        
    else:
        dataset = valid_set
        save_token = valid_token_idx
        save_char = valid_char_idx
        save_pos = valid_pos_idx
        save_ner = valid_ner_idx
        print('Processing {} validation data...'.format(len(dataset)))

    token_count = 0
    ner_count = 0

    for file in tqdm(dataset):
        with open(file, "r", encoding = "utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n','')

                if reg_idx.search(line): ## 1
                    continue

                elif line[:2]=='##' and not reg_label.search(line) : # raw_text
                    token_count+=1
                    
                    raw_text = line[3:]
                    tokenized_text, start_index = transform_source_fn(raw_text)
                    tokenized_idx = word_vocab.convert_tokens_to_idx(tokenized_text)
                    char_idx = char_vocab.convert_chars_to_idx(tokenized_text)
                    pos_idx = pos_vocab.convert_pos_to_idx(raw_text)
                    
                    save_token.append(tokenized_idx)
                    save_char.append(char_idx)
                    save_pos.append(pos_idx)
                                       
                elif line[:2]=='##' and reg_label.search(line) : # text with label
                    ner_count+=1
                    assert token_count==ner_count

                    label_text = line[3:]
                    ner_tag = transform_target_fn(label_text, tokenized_text, start_index)
                    ner_tag_to_idx = convert_entity_to_idx(ner_tag)
                    assert len(tokenized_idx)==len(ner_tag)
                    
                    save_ner.append(ner_tag_to_idx)
                    
    torch.save(save_token, './data/processed_data/{}_token_idx.pt'.format(m))
    torch.save(save_char, './data/processed_data/{}_char_idx.pt'.format(m))
    torch.save(save_pos, './data/processed_data/{}_pos_idx.pt'.format(m))
    torch.save(save_ner, './data/processed_data/{}_label.pt'.format(m))
    print('{} files saved to ./data/processed_data/{}_token_idx.pt'.format(m, m))