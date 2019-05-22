# About this fork - Still work in progress..
This fork aims at integrating [https://github.com/facebookresearch/XLM](https://github.com/facebookresearch/XLM) for performing cross-lingual bert_score.

- Download the XNLI-15 model, bpe codes and vocabulary from XLM and place them into `XLM/models/`
- add the following at the beginning of `XLM/models/vocab_xnli_15.txt`
```
<s> 101
</s> 102
<pad> 0
```
- Use `bert_score_main.py` to get similarity between two sentences.
- Use `generate_xlm_embeddings.py` to get embeddings.

## Output

```
	from bert_score import score

    refs = ['hello how are you?']
    cands = ['hola como estas?'] 
    
    refs_lang = ['en'] 
    cands_lang = ['es']

    no_idf = True if len(refs) == 1 else False

    P, R, F1 = score(cands, refs, cands_lang, refs_lang, bert="facebook-XLM", verbose=True, no_idf=no_idf)


    refs ['hello how are you?']
	cands ['hola como estas?']
	P: tensor([0.7468])
	R: tensor([0.6101])
	F1: tensor([0.6716])

```