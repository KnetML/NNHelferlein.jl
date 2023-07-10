const AMINO_ACIDS = Dict(
    "C" => 1,
    "S" => 2,
    "T" => 3,
    "A" => 4,
    "G" => 5,
    "P" => 6,
    "D" => 7,
    "E" => 8,
    "Q" => 9,
    "N" => 10,
    "H" => 11,
    "R" => 12,
    "K" => 13,
    "M" => 14,
    "I" => 15,
    "L" => 16,
    "V" => 17,
    "W" => 18,
    "Y" => 19,
    "F" => 20,
    "B" => 21,
    "Z" => 22,
    "J" => 23,
    "U" => 24,
    "X" => 25,
    "." => 26
)

const AMINO_ACIDS_REV = Dict(
    1 => "C",
    2 => "S",
    3 => "T",
    4 => "A",
    5 => "G",
    6 => "P",
    7 => "D",
    8 => "E",
    9 => "Q",
    10 => "N",
    11 => "H",
    12 => "R",
    13 => "K",
    14 => "M",
    15 => "I",
    16 => "L",
    17 => "V",
    18 => "W",
    19 => "Y",
    20 => "F",
    21 => "B",
    22 => "Z",
    23 => "J",
    24 => "U",
    25 => "X",
    26 => "."
)



"""
    aminoacid_tokenizer(sec; ignore_unknown=true)

Tokenize a protein sequence into amino acids using the following table:
    
        Amino acid | Token | Description
        --------------------------------
        C          | 1     | Cysteine
        S          | 2     | Serine
        T          | 3     | Threonine 
        A          | 4     | Alanine
        G          | 5     | Glycine
        P          | 6     | Proline
        D          | 7     | Aspartic acid
        E          | 8     | Glutamic acid
        Q          | 9     | Glutamine
        N          | 10    | Asparagine
        H          | 11    | Histidine
        R          | 12    | Arginine
        K          | 13    | Lysine
        M          | 14    | Methionine
        I          | 15    | Isoleucine
        L          | 16    | Leucine
        V          | 17    | Valine
        W          | 18    | Tryptophan
        Y          | 19    | Tyrosine
        F          | 20    | Phenylalanine

        B          | 21    | Aspartic acid or Asparagine
        Z          | 22    | Glutamic acid or Glutamine
        J          | 23    | Leucine or Isoleucine
        U          | 24    | Selenocysteine
        X          | 25    | Unknown amino acid
        .          | 26    | padding token


## Arguments:
+ `sec`: A string containing the protein sequence in uppercase or lowercase.
         All other letters or symbols will be converted to the unknwon token.
+ `ignore_unknown`: If `true`, unkown amino acids (i.e. "X") will be converted
                    to the padding token. If `false`, the embedding for "X" will
                    be trained as for all other amino acids.
"""
function aminoacid_tokenizer(sec; ignore_unknown=true)

    sec = uppercase(sec)

    sec = replace(sec, r"[^ACSTAGPDEQNHRIKMLVWYFBZJUX.]" => "X")
    if ignore_unknown
        sec = replace(sec, r"[X]" => ".")
    end

    return [AMINO_ACIDS[string(aa)] for aa in sec]
end
