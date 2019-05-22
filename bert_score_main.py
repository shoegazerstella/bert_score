from bert_score import score

if __name__ == "__main__":

    refs = ['hello how are you?']
    cands = ['hola como estas?'] 
    
    refs_lang = ['en'] 
    cands_lang = ['es']

    no_idf = True if len(refs) == 1 else False

    P, R, F1 = score(cands, refs, cands_lang, refs_lang, bert="facebook-XLM", verbose=True, no_idf=no_idf) 
    #P, R, F1 = score(cands, refs, cands_lang, refs_lang, bert="bert-base-multilingual-cased", verbose=True, no_idf=no_idf) 
    
    # bert orig
    #P, R, F1 = score(cands, refs, bert="bert-base-multilingual-cased", verbose=True, no_idf=no_idf) 
    
    print('refs', refs)
    print('cands', cands)

    print('P:', P)
    print('R:', R)
    print('F1:', F1)