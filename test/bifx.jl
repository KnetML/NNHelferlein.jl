function test_aa_tokenise()

    seq = "CSTAGPDEQNHRKMILVWYFBZJUX."
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
              11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
              21, 22, 23, 24, 26, 26]
    ignore_unk = aminoacid_tokenizer(seq) == tokens

    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
              11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
              21, 22, 23, 24, 25, 26]
    not_ignore_unk = aminoacid_tokenizer(seq, ignore_unknown=false) == tokens

    return ignore_unk && not_ignore_unk
end
