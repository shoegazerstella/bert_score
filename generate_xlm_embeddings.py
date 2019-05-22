# -*- encoding: utf-8 -*-

import os
import torch
import fastBPE

from XLM.src.utils import AttrDict
from XLM.src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from XLM.src.model.transformer import TransformerModel

vocab_path='XLM/models/vocab_xnli_15.txt'
codes_path='XLM/models/codes_xnli_15.txt'

def load_facebook_xml_model():

    print('loading facebook-XLM model..')
    # load pretrained model
    model_path = 'XLM/models/mlm_tlm_xnli15_1024.pth'
    reloaded = torch.load(model_path)
    params = AttrDict(reloaded['params'])
    #print("Supported languages: %s" % ", ".join(params.lang2id.keys()))

    # Build dictionary / update parameters / build model
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
    params.n_words = len(dico)
    params.bos_index = dico.index(BOS_WORD)
    params.eos_index = dico.index(EOS_WORD)
    params.pad_index = dico.index(PAD_WORD)
    params.unk_index = dico.index(UNK_WORD)
    params.mask_index = dico.index(MASK_WORD)

    # build model / reload weights
    model = TransformerModel(params, dico, True, True)
    model.load_state_dict(reloaded['model'])

    # get bpe
    bpe = get_bpe()

    return model, params, dico, bpe

def get_bpe():
    bpe = fastBPE.fastBPE(codes_path, vocab_path)
    return bpe

def get_vocab():
    # Dict that will contain keys and values
    dictionary =  {}
    with open(vocab_path, "r", encoding='utf-8') as f:
        for line in f:
            s = line.rstrip().split(' ')
            dictionary[s[0]] = int(s[1])
        return dictionary

def get_embeddings(model, params, dico, bpe, sentences_pairs):

    #### Get sentence representations
    # apply fastBPE
    #sentences = []

    sentences = []
    for sent in sentences_pairs:
        bpe_sent = bpe.apply([sent[0]])[0]
        lang_ = sent[1]
        sentences.append((bpe_sent, lang_))

    '''
    sentences = [
    ('the following secon@@ dary charac@@ ters also appear in the nov@@ el .', 'en'),
    ('les zones rurales offr@@ ent de petites routes , a deux voies .', 'fr'),
    ('luego del cri@@ quet , esta el futbol , el sur@@ f , entre otros .', 'es'),
    ('am 18. august 1997 wurde der astero@@ id ( 76@@ 55 ) adam@@ ries nach ihm benannt .', 'de'),
    ('اصدرت عدة افلام وث@@ اي@@ قية عن حياة السيدة في@@ روز من بينها :', 'ar'),
    ('此外 ， 松@@ 嫩 平原 上 还有 许多 小 湖泊 ， 当地 俗@@ 称 为 “ 泡@@ 子 ” 。', 'zh'),]
    ''' 

    # add </s> sentence delimiters
    sentences = [(('<s> %s </s>' % sent.strip()).split(), lang) for sent, lang in sentences]
    #sentences = [(('%s' % sent.strip()).split(), lang) for sent, lang in sentences]

    # create batch
    bs = len(sentences)
    print(bs)
    slen = max([len(sent) for sent, _ in sentences])

    word_ids = torch.LongTensor(slen, bs).fill_(params.pad_index)
    for i in range(len(sentences)):
        sent = torch.LongTensor([dico.index(w) for w in sentences[i][0]])
        word_ids[:len(sent), i] = sent

    lengths = torch.LongTensor([len(sent) for sent, _ in sentences])
    langs = torch.LongTensor([params.lang2id[lang] for _, lang in sentences]).unsqueeze(0).expand(slen, bs)

    tensor = model('fwd', x=word_ids, lengths=lengths, langs=langs, causal=False).contiguous()
    '''
    The variable tensor is of shape (sequence_length, batch_size, model_dimension).
    tensor[0] is a tensor of shape (batch_size, model_dimension) 
    that corresponds to the first hidden state of the last layer of each sentence.
    '''
    tensor = tensor.transpose(0, 1)

    return tensor
    

if __name__ == "__main__":

    sentences_dict = {
        'en' : 'hello how are you?',
        'es' : 'hola como estas?'
    }

    model, params, dico, bpe = load_facebook_xml_model()
    tensor = get_embeddings(model, params, dico, bpe, sentences_dict)
    print(tensor.size())