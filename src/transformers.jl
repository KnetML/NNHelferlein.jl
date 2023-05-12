#
# Helpers and definitions for transformer networks:
#



"""
    function positional_encoding_sincos(n_embed, n_seq)

Calculate and return a matrix of size `[n_embed, n_seq]` of
positional encoding values
following the sin and cos style in the paper
*Vaswani, A. et al.; Attention Is All You Need;
31st Conference on Neural Information Processing Systems (NIPS 2017),
Long Beach, CA, USA, 2017.*
"""
function positional_encoding_sincos(n_embed, n_seq)

    angl = [1/(10000^(2*i/n_embed)) for i in 1:n_embed/2]
    angl = angl * permutedims(1:n_seq)
    pos_enc = vcat(sin.(angl), cos.(angl))
    return convert2KnetArray(pos_enc)
end



"""
    struct PositionalEncoding <: AbstractLayer

Positional encoding layer. Only *sincos*-style (according to
Vaswani, et al., NIPS 2017) is implemented.

The layer takes an array of any number of dimensions (>=2), calculates
the Vaswani-2017-style positional encoding and adds the encoding to each plane
of the array.
"""
struct PositionalEncoding <: AbstractLayer
    style
    PositionalEncoding(;style=:sincos) = new(style)
end

function (l::PositionalEncoding)(x)
    # only one style implemented yet:
    # if l.style == sincos
        x = convert2KnetArray(x) .+ positional_encoding_sincos(size(x)[1], size(x)[2])
        return x
end



"""
    function mk_padding_mask(x; pad=TOKEN_PAD, add_dims=false)

Make a padding mask; i.e. return an Array of type
`KnetArray{Float32}` (or `Array{Float32}`) similar to `x` but with
two additional dimensions of size 1 in the middle (this will represent the
2nd seq_len and the number of heads) in multi-head attention
and the
value `1.0` at each position where `x` is `pad` and `0.0` otherwise.

The function can be used for creating padding masks for attention
mechanisms.

### Arguments:
+ `x`: Array of sequences (typically a matrix with n_cols sequences
    of length n_rows)
+ `pad`: value for the token to be masked
+ `add_dims`: if `true`, 2 additional dimensions are inserted to 
    return a 4-D-array as needed for transformer architectures. Otherwise
    the size of the returned array is similar to `x`.
"""
function mk_padding_mask(x; pad=TOKEN_PAD, add_dims=false)

    if add_dims
       return reshape(convert2KnetArray(x .== pad), size(x)[1],1,1,size(x)[2])
    else
       return convert2KnetArray(x .== pad)
    end
end


"""
    function mk_peek_ahead_mask(x; dim=1)
    function mk_peek_ahead_mask(n_seq)

Return a matrix of size `[n_seq, n_seq]` filled with 1.0 and the *uppper triangle*
set to 0.0.
Type is `CuArray{Float32}` in GPU context, `Array{Float32}` otherwise.
The matrix can be used as peek-ahead mask in transformers.

`dim=1` specifies the dimension in which the sequence length is
represented. For un-embedded data this is normally `1`, i.e. the
shape of `x` is [n_seq, n_mb]. After embedding the shape probably is
[depth, n_seq, n_mb].
"""
function mk_peek_ahead_mask(x::AbstractArray; dim=1)

    n_seq = size(x)[dim]
    return mk_peek_ahead_mask(n_seq)
end

function mk_peek_ahead_mask(n_seq::Int)

    return ifgpu(1 .- UpperTriangular(ones(n_seq, n_seq)))
end



"""
    function dot_prod_attn(q, k, v; mask=nothing)

Generic scaled dot product attention following the paper of
Vaswani et al., (2017), *Attention Is All You Need*.

### Arguments:
+ `q`: query of size `[depth, n_seq_q, ...]`
+ `k`: key of size `[depth, n_seq_v, ...]`
+ `v`: value of size `[depth, n_seq_v, ...]`
+ `mask`: mask for attention factors may have different shapes but must be
        broadcastable for addition to the scores tensor (which as the same size as
        alpha `[n_seq_v, n_seq_q, ...]`). In transformer context typical masks are one of:
        padding mask of size `[n_seq_v, ...]` or a peek-ahead mask of size `[n_seq_v, n_seq_v]`
        (which is only possible in case of self-attention when all sequence lengths
        are identical).

`q, k, v` must have matching leading dimensions (i.e. same depth or embedding).
`k` and `v` must have the same sequence length.

### Return values:
+ `c`: context as alpha-weighted sum of values with size [depth, n_seq_v, ...]
+ `alpha`: attention factors of size [n_seq_v, n_seq_q, ...]
"""
function dot_prod_attn(q, k, v; mask=nothing)

    score = bmm(k, q, transA=true) ./ Float32(sqrt(size(k)[1]))  # [s_v x s_k x mb]

    if !isnothing(mask)
        score = score .+ mask * Float32(-1e9)
    end

    α = softmax(score, dims=1)
    c = bmm(v, α)
    return c, α
end


"""
    struct MultiHeadAttn <: AbstractLayer

Multi-headed attention layer, designed following the Vaswani, 2017 paper.

### Constructor:

    MultiHeadAttn(depth, n_heads) 

+ `depth`: Embedding depth
+ `n_heads`: number of heads for the attention.

### Signature:

    function(mha::MultiHeadAttn)(q, k, v; mask=nothing)

`q, k, v` are 3-dimensional tensors of the same size
([depth, seq_len, n_minibatch]) and the optional mask must be of 
size [seq_len, n_minibatch] and mark masked positions with 1.0.

"""
mutable struct MultiHeadAttn
    dense_q        # x -> q
    dense_k        # x -> K
    dense_v        # x -> v
    depth          # embedding
    n_heads        #
    h_depth        # embedding / heads
    dense_out      # out layer
    MultiHeadAttn(depth, n_heads) = new(Linear(depth, depth), Linear(depth, depth), 
                                        Linear(depth, depth),
                                        depth, n_heads, depth÷n_heads,
                                        Linear(depth, depth))
end

function(mha::MultiHeadAttn)(q, k, v; mask=nothing)

    q = mha.dense_q(q)      # [depth, n_seq, n_mb]
    k = mha.dense_k(k)
    v = mha.dense_v(v)

    q = separate_heads(q, mha.n_heads)      # [depth/n, n_seq, n_heads, n_mb]
    k = separate_heads(k, mha.n_heads)
    v = separate_heads(v, mha.n_heads)

    c, α = dot_prod_attn(q, k, v, mask=mask)  # c: [depth/n, n_seq, n_heads, n_mb]
                                              # α: [n_seq, n_seq, n_heads, n_mb]
    c = merge_heads(c)                        # [depth, n_seq_ n_mb]
    return mha.dense_out(c), α
end




"""
    function separate_heads(x, n)

Helper function for multi-headed attention mechanisms: 
an additional second dimension is added to a tensor of minibatches
by splitting the first (i.e. depth).
"""
function separate_heads(x, n)
    depth, seq, mb = size(x)
    mh_depth = depth ÷ n
    x = reshape(x, mh_depth, n, :, mb)     # split depth in 2 separate dims for heads
    return permutedims(x, (1,3,2,4))       # bring seq-len back to 2nd dim
end

"""
    function merge_heads(x)

Helper to merge the result of multi-headed attention back to full
depth .
"""
function merge_heads(x)
    mh_depth, seq, n, mb = size(x)
    depth = mh_depth * n
    x = permutedims(x, (1,3,2,4))          # bring heads back to 2nd dim
    return reshape(x, depth, :, mb)        # merde depth and heads (dims 1 and 2) into 1st dim
end




# """
#     struct BertEncoder
# 
# """
# 
"""
    TFEncoderLayer

A Bert-like encoder layer to be used as part of a Bert-like
transformer.
The layer consists of a multi-head attention sub-layer followed by
a feed-forward network of size depth -> 4*depth -> depth. 
Both parts have separate residual connections and layer normalisation.

The design follows the original paper "Attention is all you need" 
by Vaswani, 2017.

### Constructor:

    TFEncoderLayer(depth, n_heads, drop)

+ `depth`: Embedding depth
+ `n_heads`: number of heads for the multi-head attention
+ `drop_rate`: dropout rate

### Signature:

    (el::TFEncoderLayer)(x; mask=nothing)

Objects of type `TFEncoderLayer` are callable and expect a 
3-dimensional array of size [embedding_depth, seq_len, minibatch_size] 
as input. 
The optional `mask` must be of size [seq_len, minibatch_size] and
mark masked positions with 1.0.

It returns a tensor of the same size as the input and the self-attention
factors of size [seq_len, seq_len, minibatch_size].
"""
mutable struct TFEncoderLayer
    mha           # multi-head attn
    drop1
    norm1         # layer-norm
    ffwd1; ffwd2  # final feed forward 4*depth
    drop2
    norm2

    TFEncoderLayer(depth, n_heads, drop_rate) = new(MultiHeadAttn(depth, n_heads),
                                                Dropout(drop_rate),
                                                LayerNorm(depth),
                                                Linear(depth, depth*4, actf=relu),
                                                Linear(depth*4, depth),
                                                Dropout(drop_rate),
                                                LayerNorm(depth)
                                                )
    end

function (el::TFEncoderLayer)(x; mask=nothing)

    c, α = el.mha(x, x, x, mask=mask)   # always: [depth, seq_len, mb_size]
    c = el.drop1(c)
    o1 = el.norm1(x .+ c)

    o2 = el.ffwd1(o1)                   # [depth*4, seq_len, mb_size]
    o2 = el.ffwd2(o2)                   # back: [depth, seq_len, mb_size]
    o2 = el.drop2(o2)
    o2 = el.norm2(o1 .+ o2)
    return o2, α
end

"""
    TFEncoder

A Bert-like encoder to be used as part of a tranformer. 
The encoder is build as a stack of `TFEncoderLayer`s 
which is entered after embedding, positional encoding and
generation of a padding mask.

### Constructor:
    
    TFEncoder(n_layers, depth, n_heads; drop_rate=0.1)

### Signature:

    (e::TFEncoder)(x)

The encoder is called with a matrix of embedded tokens of size
`[depth, seq_len, n_minibatch]` and returns a tensor of size
`[depth, seq_len, n_minibatch]`.
"""
mutable struct TFEncoder
    depth          # embeding depth
    n_layers
    layers         # list of actual encoder layers
    drop           # dropout layer

    TFEncoder(n_layers, depth, n_heads; drop_rate=0.1) =
            new(depth,
                n_layers,
                [TFEncoderLayer(depth, n_heads, drop_rate) for i in 1:n_layers],
                Dropout(drop_rate))
end

# x is [depth, seq_len, mb_size]
# mask is the padding mask of size [seq_len, mb_size] with 1.0 at masked positions
#
function (e::TFEncoder)(x; mask=nothing)

    n_seq = size(x)[2]
    x = x .+ positional_encoding_sincos(e.depth, n_seq)
    x = e.drop(x)

    for l in e.layers
        x,α = l(x, mask=mask)
    end
    return x
end


# 
# 
# """
#     struct BertDecoder
# 
# """
"""
    TFDecoderLayer

A Bert-like decoder layer to be used as part of a Bert-like
transformer.
The layer consists of a multi-head self-attention sub-layer,
a multi-head attention sub-layer followed by
a feed-forward network of size depth -> 4*depth -> depth. 
All three parts have separate residual connections and layer normalisation.

The design follows the original paper "Attention is all you need" 
by Vaswani, 2017.

### Constructor:

    TFDecoderLayer(depth, n_heads, drop)

+ `depth`: Embedding depth
+ `n_heads`: number of heads for the multi-head attention
+ `drop`: dropout rate

### Signature:

    (el::TFDecoderLayer)(x, h_encoder; enc_m_pad=nothing, m_combi=nothing)

Objects of type `TFDecoderLayer` are callable and expect a 
minibatch of embedded sequences as input.
+ `x`: 3-dimensional array of size [embedding_depth, seq_len, minibatch_size] 
+ `h_encoder`: output of the encoder stack
+ `enc_m_pad`: optional padding mask for the encoder output
+ `m_combi`: optional mask for the decoder self-attention
             combining padding and peek-ahead mask.


It returns a tensor of the same size as the input, the self-attention
factors and the decoder-encoder attention factors.
"""
mutable struct TFDecoderLayer
    mhsa          # multi-head attn
    drop1
    norm1         # layer-norm
    mha
    drop2
    norm2
    ffwd1; ffwd2  # final feed forward
    drop3
    norm3

    TFDecoderLayer(depth, n_heads; drop_rate=0.1) = new(MultiHeadAttn(depth, n_heads),
                                             Dropout(drop_rate),
                                             LayerNorm(depth),
                                             MultiHeadAttn(depth, n_heads),
                                             Dropout(drop_rate),
                                             LayerNorm(depth),
                                             Linear(depth, depth*4, actf=relu),
                                             Linear(depth*4, depth),
                                             Dropout(drop_rate),
                                             LayerNorm(depth)
                                             )
    end

function (dec::TFDecoderLayer)(x, h_encoder; enc_mask=nothing, dec_mask=nothing)

    self_c, α1 = dec.mhsa(x, x, x, mask=dec_mask)    # c: [depth, seq_len, mb_size]
                                                    # α: [seq1_len, seq2_len, n_heads, mb_size]
    self_c = dec.drop1(self_c)
    self_c = dec.norm1(x .+ self_c)

    o1, α2 = dec.mha(self_c, h_encoder, h_encoder, mask=enc_mask)
    o1 = dec.drop2(o1)
    o1 = dec.norm2(o1 .+ self_c)

    o2 = dec.ffwd1(o1)
    o2 = dec.ffwd2(o2)
    o2 = dec.drop2(o2)
    o2 = dec.norm2(o1 + o2)
    return o2, α1, α2
end





"""
    TFDecoder

A Bert-like decoder to be used as part of a tranformer. 
The decoder is build as a stack of `TFDecoderLayer`s 
which is entered after embedding, positional encoding and
generation of a padding mask and a peek-ahead mask.

### Constructor:
    
    TFDecoder(n_layers, depth, n_heads, vocab_size; 
              pad_id=NNHelferlein.TOKEN_PAD, drop_rate=0.1)

### Signature:

    (e::TFdecoder)(x)

The decoder is called with a matrix of token ids of size
`[seq_len, n_minibatch]` and returns a tensor of size
`[depth, seq_len, n_minibatch]` and the attention factors.
"""
mutable struct TFDecoder
    depth          # embeding depth
    n_layers
    layers         # list of actual encoder layers
    drop           # dropout layer

    TFDecoder(n_layers, depth, n_heads; drop_rate=0.1) =
                new(depth,
                n_layers,
                [TFDecoderLayer(depth, n_heads, drop_rate=drop_rate) for i in 1:n_layers],
                Dropout(drop_rate))
end

# y is [depth, seq_len, mb_size]
# h_enc is encoder output
# enc_mask is the padding mask of enoder input (x) with 1.0 at masked positions
# dec_mask is the padding mask of decoder input (y) with 1.0 at masked positionso#
#
function (d::TFDecoder)(y, h_enc; enc_mask=nothing, dec_mask=nothing)

    n_seq = size(y)[2]
    peek_mask = mk_peek_ahead_mask(n_seq)                           
    if isnothing(dec_mask)
        dec_mask = zeros(Float32, n_seq, n_seq) |> ifgpu
    end
    combi_mask = max.(peek_mask, dec_mask)                            # combined target sequence mask needed
                                                             # for self-attn
    y = y .+ positional_encoding_sincos(d.depth, n_seq)
    y = d.drop(y)

    α2 = init0(0,0,0)
    for l in d.layers
        y,α1,α2 = l(y, h_enc, enc_mask=enc_mask, dec_mask=combi_mask)
    end
    return y, α2
end


"""
    mutable struct Transformer

A Bert-like transformer network consisting of an encoder and a decoder
stack.

### Constructor:

    Transformer(n_layers, depth, heads; drop_rate=0.1)

+ `n_layers`: number of layers in encoder and decoder
+ `depth`: embedding depth
+ `heads`: number of heads for the multi-head attention
+ `drop_rate`: dropout rate used in all layers

### Signature:
    
    (tf::Transformer)(x, y; enc_mask=nothing, dec_mask=nothing)

The transformer is called with two 3-d-arrays of embedded sequences
`x` and `y` of size `[depth, seq_len, n_minibatch]` and returns a tensor of size
`[depth, seq_len_y, n_minibatch]`. 
Sequences `x` and `y` may be of different lengths; output has always the
same dimensions as `y`.

Attention factors of the last rund 
are stored in the field `α` of the transformer object.

`enc_mask` and `dec_mask` are optional padding masks for the encoder
and decoder input, respectively. They must be of size `[seq_len, n_minibatch]`.
"""
mutable struct Transformer
    n_layers
    depth
    encoder
    decoder
    α       # attn-matrix dec to enc of the last return

    Transformer(n_layers, depth, heads; drop_rate=0.1) = 
        new(n_layers,
            depth,
            TFEncoder(n_layers, depth, heads, drop_rate=drop_rate),
            TFDecoder(n_layers, depth, heads, drop_rate=drop_rate),
            nothing)
end


function (tf::Transformer)(x, y; enc_mask=nothing, dec_mask=nothing)

    h_enc = tf.encoder(x, mask=enc_mask)
    y, tf.α = tf.decoder(y, h_enc, dec_mask=dec_mask, enc_mask=enc_mask)
    return y
end


"""
    mutable struct TokenTransformer

A wrapper around the `Transformer` object that takes sequences of
token ids as input.

### Constructor:

    TokenTransformer(n_layers, depth, heads, 
                     x_vocab, y_vocab;
                     drop_rate=0.1)

+ `n_layers`: number of layers in encoder and decoder
+ `depth`: embedding depth
+ `heads`: number of heads for the multi-head attention
+ `x_vocab`: vocabulary size of the input sequences as integer value
             or a `WordTokenizer` object
+ `y_vocab`: vocabulary size of the output sequences as integer value
                or a `WordTokenizer` object
+ `drop_rate`: dropout rate used in all layers

### Signature:
        
        (tt::TokenTransformer)(x, y; enc_mask=nothing, dec_mask=nothing
                               de_embed=false)

The transformer is called with two 2-d-arrays of token ids
`x` and `y` of size `[seq_len, n_minibatch]` which may be of 
different lengths. It returns a tensor of size
`[y_vocab, seq_len_y, n_minibatch]` with the raw activations 
of output neurons or, if
`de_embed` is set to `true`, a 2-d-array of size
`[seq_len_y, n_minibatch]` with the sequences of generated tokens.
"""
mutable struct TokenTransformer
    n_layers
    depth
    embed_enc
    embed_dec
    predict
    transformer
    x_len        # number of vocab items
    y_len
    α            # attn-matrix dec to enc of the last return

    function TokenTransformer(n_layers, depth, heads,
                              x_vocab, y_vocab;
                              drop_rate=0.1)

        if x_vocab isa WordTokenizer
            x_len = length(x_vocab)
        else
            x_len = x_vocab
        end
        if y_vocab isa WordTokenizer
            y_len = length(y_vocab)
        else
            y_len = y_vocab
        end

        return new(n_layers, depth,
            Embed(x_len, depth),
            Embed(y_len, depth),
            Linear(depth, y_len),
            Transformer(n_layers, depth, heads, drop_rate=drop_rate),
            x_len, 
            y_len, 
            nothing)
    end
end

function (tt::TokenTransformer)(x, y; enc_mask=nothing, dec_mask=nothing,
                                embedded=true)

    x = tt.embed_enc(x)
    y = tt.embed_enc(y)

    y = tt.transformer(x, y, enc_mask=enc_mask, dec_mask=dec_mask)
    y = tt.predict(y)
    tt.α = tt.transformer.α
 
    if !embedded
        y = de_embed(y, remove_dim=true)
    end
    return y
end

# function fo flip 1 and zero in a array:
