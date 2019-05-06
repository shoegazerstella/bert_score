from bert_score import score

if __name__ == "__main__":

    cands = ['hello how are you?']
    refs = ['hola como estas?']

    cands_lang = ['en']
    refs_lang = ['es']

    no_idf = True if len(cands) == 1 else False

    P, R, F1 = score(cands, refs, cands_lang, refs_lang, bert="facebook-XLM", verbose=True, no_idf=no_idf) 
    #P, R, F1 = score(cands, refs, cands_lang, refs_lang, bert="bert-base-multilingual-cased", verbose=True, no_idf=no_idf) 

    print(cands)
    print(refs)

    print('P:', P)
    print('R:', R)
    print('F1:', F1)