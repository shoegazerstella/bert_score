#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
from math import log
from itertools import chain
from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial
from tqdm.auto import tqdm
import generate_xlm_embeddings as xlm_emb
from sacremoses import MosesTokenizer, MosesPunctNormalizer

__all__ = ['bert_types']

bert_types = [
    'bert-base-uncased',
    'bert-large-uncased',
    'bert-base-cased',
    'bert-large-cased',
    'bert-base-multilingual-uncased',
    'bert-base-multilingual-cased',
    'bert-base-chinese',
    'facebook-XLM'
]


def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        print(i, a)
        padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, :lens[i]] = 1
    
    return padded, lens, mask


def bert_encode(model, x, attention_mask):
    model.eval()
    x_seg = torch.zeros_like(x, dtype=torch.long)
    with torch.no_grad():
        x_encoded_layers, pooled_output = model(x, x_seg, attention_mask=attention_mask, output_all_encoded_layers=False)
    return x_encoded_layers


def process(a, tokenizer=None, XLM=False):
    if not tokenizer is None:
        if XLM:
            bpe = xlm_emb.get_bpe()
            a = bpe.apply([a])
            a = [('<s> %s </s>' % sent.strip()).split() for sent in a]
            #a = [('%s' % sent.strip()).split() for sent in a]
            a = convert_tokens_to_ids(a[0])
        else:
            a = ["[CLS]"]+tokenizer.tokenize(a)+["[SEP]"]
            a = tokenizer.convert_tokens_to_ids(a)
    return set(a)


def get_idf_dict(arr, tokenizer, nthreads=4, XLM=False):
    """
    Returns mapping from word piece index to its inverse document frequency.
    Args:
        - :param: `arr` (list of str) : sentences to process.
        - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
        - :param: `nthreads` (int) : number of CPU threads to use
    """

    idf_count = Counter()
    num_docs = len(arr)

    process_partial = partial(process, tokenizer=tokenizer, XLM=XLM)

    with Pool(nthreads) as p:
        idf_count.update(chain.from_iterable(p.map(process_partial, arr)))

    idf_dict = defaultdict(lambda : log((num_docs+1)/(1)))
    idf_dict.update({idx:log((num_docs+1)/(c+1)) for (idx, c) in idf_count.items()})

    return idf_dict


def convert_tokens_to_ids(tokens, max_len=None):
    # https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization.py
    """Converts a sequence of tokens into ids using the vocab."""
    
    max_len = max_len if max_len is not None else int(1e12)
    vocab = xlm_emb.get_vocab()

    # remove spaces in tokens, example: ['ho@@ la', 'como', 'estas']
    #tokens = [i.split(' ') for i in tokens]
    # flatten
    #tokens = [item for sublist in tokens for item in sublist]

    ids = []
    for token in tokens:
        ids.append(vocab[token])
    if len(ids) > max_len:
        logger.warning(
            "Token indices sequence length is longer than the specified maximum "
            " sequence length for this BERT model ({} > {}). Running this"
            " sequence through BERT will result in indexing errors".format(len(ids), max_len)
        )
    return ids


def collate_idf(arr, tokenizer, numericalize, idf_dict,
                pad="[PAD]", device='cuda:0', XLM=False):

    """
    Helper function that pads a list of sentences to have the same length and
    loads idf score for words in the sentences.
    Args:
        - :param: `arr` (list of str): sentences to process.
        - :param: `tokenize` : a function that takes a string and return list
                  of tokens.
        - :param: `numericalize` : a function that takes a list of tokens and
                  return list of token indexes.
        - :param: `idf_dict` (dict): mapping a word piece index to its
                               inverse document frequency
        - :param: `pad` (str): the padding token.
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """
    if XLM:
        bpe = xlm_emb.get_bpe()
        arr = [bpe.apply([a]) for a in arr]
        arr = [('<s> %s </s>' % a[0].strip()).split() for a in arr]
        #arr = [('%s' % a[0].strip()).split() for a in arr]
        arr = [numericalize(a) for a in arr]
    else:
        arr = [["[CLS]"]+tokenizer(a)+["[SEP]"] for a in arr]
        arr = [numericalize(a) for a in arr]

    idf_weights = [[idf_dict[i] for i in a] for a in arr]

    if XLM:
        pad_token = numericalize(['<pad>'])[0]
    else:
        pad_token = numericalize([pad])[0]

    print('*****')
    print(arr)
    print('idf_dict', idf_dict)
    print('idf_weights', idf_weights)
    print('*****')
    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
    padded_idf, _, _ = padding(idf_weights, pad_token, dtype=torch.float)

    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)

    return padded, padded_idf, lens, mask


def get_bert_embedding(all_sens, model, tokenizer, idf_dict,
                       batch_size=-1, device='cuda:0'):
    """
    Compute BERT embedding in batches.
    Args:
        - :param: `all_sens` (list of str) : sentences to encode.
        - :param: `model` : a BERT model from `pytorch_pretrained_bert`.
        - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
        - :param: `idf_dict` (dict) : mapping a word piece index to its
                               inverse document frequency
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """

    padded_sens, padded_idf, lens, mask = collate_idf(all_sens,
                                                      tokenizer.tokenize, tokenizer.convert_tokens_to_ids,
                                                      idf_dict,
                                                      device=device)

    if batch_size == -1: batch_size = len(all_sens)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_sens), batch_size):
            batch_embedding = bert_encode(model, padded_sens[i:i+batch_size],
                                          attention_mask=mask[i:i+batch_size])
            # batch_embedding = torch.stack(batch_embedding)
            embeddings.append(batch_embedding)
            del batch_embedding

    total_embedding = torch.cat(embeddings, dim=0)

    return total_embedding, lens, mask, padded_idf


def get_bert_embedding_xlm(all_sens, lang, model, params, dico, bpe, tokenizer, idf_dict,
                       batch_size=-1, device='cuda:0'):
    """
    Compute BERT embedding in batches.
    Args:
        - :param: `all_sens` (list of str) : sentences to encode.
        - :param: `model` : a BERT model from `pytorch_pretrained_bert`.
        - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
        - :param: `idf_dict` (dict) : mapping a word piece index to its
                               inverse document frequency
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """
    padded_sens, padded_idf, lens, mask = collate_idf(all_sens,
                                                      tokenizer, convert_tokens_to_ids,
                                                      idf_dict,
                                                      device=device,
                                                      XLM=True)

    # get embedding
    sentences = []
    for i in range(0, len(all_sens)):
        sentences.append((all_sens[i], lang[i]))
    total_embedding = xlm_emb.get_embeddings(model, params, dico, bpe, sentences)

    return total_embedding, lens, mask, padded_idf


def greedy_cos_idf(ref_embedding, ref_lens, ref_masks, ref_idf,
                   hyp_embedding, hyp_lens, hyp_masks, hyp_idf):
    """
    Compute greedy matching based on cosine similarity.
    Args:
        - :param: `ref_embedding` (torch.Tensor):
                   embeddings of reference sentences, BxKxd,
                   B: batch size, K: longest length, d: bert dimenison
        - :param: `ref_lens` (list of int): list of reference sentence length.
        - :param: `ref_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                   reference sentences.
        - :param: `ref_idf` (torch.Tensor): BxK, idf score of each word
                   piece in the reference setence
        - :param: `hyp_embedding` (torch.Tensor):
                   embeddings of candidate sentences, BxKxd,
                   B: batch size, K: longest length, d: bert dimenison
        - :param: `hyp_lens` (list of int): list of candidate sentence length.
        - :param: `hyp_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                   candidate sentences.
        - :param: `hyp_idf` (torch.Tensor): BxK, idf score of each word
                   piece in the candidate setence
    """

    ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

    batch_size = ref_embedding.size(0)

    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
    
    masks = masks.expand(batch_size, masks.size(1), masks.size(2)).contiguous().view_as(sim)

    masks = masks.float().to(sim.device)
    sim = sim * masks

    word_precision = sim.max(dim=2)[0]
    word_recall = sim.max(dim=1)[0]

    hyp_idf.div_(hyp_idf.sum(dim=1, keepdim=True))
    ref_idf.div_(ref_idf.sum(dim=1, keepdim=True))

    precision_scale = hyp_idf.to(word_precision.device)
    recall_scale = ref_idf.to(word_recall.device)
    
    P = (word_precision * precision_scale).sum(dim=1)
    R = (word_recall * recall_scale).sum(dim=1)
    
    F = 2 * P * R / (P + R)
    return P, R, F


def bert_cos_score_idf(model, refs, hyps, refs_lang, hyps_lang, tokenizer, idf_dict, bert,
                       verbose=False, batch_size=256, device='cuda:0'):
    """
    Compute BERTScore.
    Args:
        - :param: `model` : a BERT model in `pytorch_pretrained_bert`
        - :param: `refs` (list of str): reference sentences
        - :param: `hyps` (list of str): candidate sentences
        - :param: `tokenzier` : a BERT tokenizer corresponds to `model`
        - :param: `idf_dict` : a dictionary mapping a word piece index to its
                               inverse document frequency
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `batch_size` (int): bert score processing batch size
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """

    if bert == 'facebook-XLM':
        model, params, dico, bpe = xlm_emb.load_facebook_xml_model()

    preds = []
    iter_range = range(0, len(refs), batch_size)
    if verbose: iter_range = tqdm(iter_range)

    for batch_start in iter_range:

        batch_refs = refs[batch_start:batch_start+batch_size]
        batch_hyps = hyps[batch_start:batch_start+batch_size]

        if bert == 'facebook-XLM':
            batch_lang_refs = refs_lang[batch_start:batch_start+batch_size]
            batch_lang_hyps = hyps_lang[batch_start:batch_start+batch_size]

            # get bert embeddings

            with torch.no_grad():
                ref_stats = get_bert_embedding_xlm(batch_refs, batch_lang_refs, model, params, dico, bpe, tokenizer, idf_dict,
                                        device=device)

                hyp_stats = get_bert_embedding_xlm(batch_hyps, batch_lang_hyps, model, params, dico, bpe, tokenizer, idf_dict,
                                            device=device)
                
        else:
            ref_stats = get_bert_embedding(batch_refs, model, tokenizer, idf_dict,
                                        device=device)

            hyp_stats = get_bert_embedding(batch_hyps, model, tokenizer, idf_dict,
                                        device=device)

        print('***************************')
        print('\nref_stats')
        print('ref_stats[0].size()', ref_stats[0].size())
        print('ref_stats[1]', ref_stats[1].size(), ref_stats[1])
        print('ref_stats[2]', ref_stats[2].size(), ref_stats[2])
        print('ref_stats[3]', ref_stats[3].size(), ref_stats[3])

        print('\nhyp_stats')
        print('hyp_stats[0].size()', hyp_stats[0].size())
        print('hyp_stats[1]', hyp_stats[1].size(), hyp_stats[1])
        print('hyp_stats[2]', hyp_stats[2].size(), hyp_stats[2])
        print('hyp_stats[3]', hyp_stats[3].size(), hyp_stats[3])
        print('***************************')

        P, R, F1 = greedy_cos_idf(*ref_stats, *hyp_stats)
        preds.append(torch.stack((P, R, F1), dim=1).cpu())

    preds = torch.cat(preds, dim=0)
    return preds
