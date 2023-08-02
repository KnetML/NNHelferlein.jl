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




# Aminoacid encoding:
#
const BIFX_DIR = "bifx"
const BLOSUM62_FILE = "blosum62.csv"
const VHSE8_FILE = "vhse8.csv"

# tbl is a table of size depthx21 with the AS embeddinge one per column.
# the last column is just zeros and is used for all tokens > 20 and < 1.
#
function embed_aminoacids(x::AbstractArray, tbl)
    x[x .< 1] .= 21
    x[x .> 20] .= 21
    return tbl[:,x] 
end

function embed_aminoacids(x::Int, tbl)
    if x < 1 || x > 20
        return tbl[:,21]
    else
        return tbl[:,x]
    end
end


function read_embedding_matrix(file)
    df = dataframe_read(joinpath(DATA_DIR, BIFX_DIR, file))
    tokens = nrow(df)
    depth = ncol(df) - 1

    tbl = df[1:tokens, 2:depth+1] |> Matrix 
    tbl = permutedims(tbl, (2,1))
    tbl = hcat(tbl, zeros(depth,1)) |> ifgpu
    return tbl
end






"""
    embed_blosum62(x)

Embed a protein sequence into a 21-dimensional vector using the BLOSUM62
amino acid substitution matrix. Aminoacid are encoded as with 
*NNHelferleins* `aminoacid tokenizer` function.
`x` can be any `AbstractArray` of `Int` and a dimension of size 21 will be
added as the first dimension. 
"""
function embed_blosum62(x)

    tbl = read_embedding_matrix(BLOSUM62_FILE)
    return embed_aminoacids(x, tbl)
end

"""
    embed_vhse8(x)

Embed a protein sequence into a 8-dimensional vector using the VHSE8
amino acid embedding scheme. Aminoacid are encoded as with 
*NNHelferleins* `aminoacid tokenizer` function.
`x` can be any `AbstractArray` of `Int` and a dimension of size 21 will be
added as the first dimension. 
"""
function embed_vhse8(x)

    tbl = read_embedding_matrix(VHSE8_FILE)
    return embed_aminoacids(x, tbl)
end




"""
    EmbedAminoAcids <: AbstractLayer

Embed a protein sequence into a 21-dimensional vector using the BLOSUM62
amino acid substitution matrix
or as a 8-dimensional vector using the VHSE8 parameters.
Aminoacids must be encoded acording to 
*NNHelferlein's* `aminoacid tokenizer` function.

Layer input a is a n-dimensional array of an Integer type. Output is a
(n+1)-dimensional array of Float32 type with a first (added) dimension 
of size 21 or 8.

## Constructor:
+ `EmbedAminoAcids(embedding::Symbol=:blosum62)`: 
    + `embedding=:blosum62`: Either `:blosum62` or `:vhse8` 
                to select the embedding scheme.
"""
struct EmbedAminoAcids <: AbstractLayer
    scheme
    tbl::AbstractArray
    function EmbedAminoAcids(scheme=:blosum62)
        if scheme == :blosum62
            tbl = read_embedding_matrix(BLOSUM62_FILE)
            return new(scheme, tbl)
        elseif scheme == :vhse8
            tbl = read_embedding_matrix(VHSE8_FILE)
            return new(scheme, tbl)
        else
            error("Unknown embedding scheme: $scheme")
        end
    end
end

(e::EmbedAminoAcids)(x) = embed_aminoacids(x, e.tbl)

function Base.summary(l::EmbedAminoAcids; indent=0)
    n = get_n_params(l)
    if l.scheme == :blosum62
        s1 = "Blosum62 embeding layer"
        i,o = 21, 20
    elseif l.scheme == :vhse8
        s1 = "VHSE8 embeding layer"
        i,o = 21, 8
    else
        s1 = "unknown amino acid embeding layer"
    end
    s = "$s1 $i tokens → depth $o,"

    println(print_summary_line(indent, s, n))
    return 1
end

