# test attn:
#
using Knet, NNHelferlein

depth, mb, t = 3, 4, 6

h_enc = convert2KnetArray(randn(Float32, depth, mb, t))
h_t = convert2KnetArray(randn(Float32, depth, mb))

function test_attn_reset()
    a = AttnBahdanau(depth, depth)

    a.projections = rand(depth, depth)
    a()   # reset = true

    return !isnothing(a.projections)
end

function test_attn(attn)
    a = attn(depth, depth)
    c,α = a(h_t, h_enc)

    return size(c) == (3,4)
end

function test_attnDot()
    a = AttnDot()
    c,α = a(h_t, h_enc)

    return size(c) == (3,4)
end

function test_attnLocation()
    a = AttnLocation(t, depth)
    c,α = a(h_t, h_enc)

    return size(c) == (3,4)
end

function test_attnInFeed()
    a = AttnInFeed(t, depth, depth)
    c,α = a(h_t, init0(depth), h_enc)

    return size(c) == (3,4)
end


import NNHelferlein: resize_attn_mask
function test_attn_resize()
    size_3 = size(resize_attn_mask(rand(5,5,5))) # 5,5,5
    size_2 = size(resize_attn_mask(rand(5,5))) # 1,5,5
    size_1 = size(resize_attn_mask(rand(5))) # 1,1,5
    size_o = size(resize_attn_mask(rand(5,5,5,5))) # 1,

    return size_3 == (5,5,5) &&
           size_2 == (1,5,5) &&
           size_1 == (1,1,5) &&
           size_o == (1,)
end
    


# transformer tests:
#
function test_dpa()
    kqv = rand(Float32, 3,8,10)
    c,a = dot_prod_attn(kqv, kqv, kqv)

    return size(c) == (3,8,10) && size(a) == (8,8,10)
end

function test_masks()
    seqs = rand(1:16, 4,6)
    el = Embed(16,8)
    seqs_e = el(seqs)

    pl = PositionalEncoding()
    pos_enc = pl(seqs)   # assert 4x6

    peek_ah = mk_peek_ahead_mask(seqs)
    padd = mk_padding_mask(seqs, add_dims=true)

    return size(pos_enc) == (4,6) &&
           size(peek_ah) == (4,4) &&
           size(padd) == (4,1,1,6)
end


function test_dotp_attn()

    function separate_heads(x, n)
        depth, seq, mb = size(x)
        mh_depth = depth ÷ n
        x = reshape(x, mh_depth, n, :, mb)     # split depth in 2 separate dims for heads
        return permutedims(x, (1,3,2,4))       # bring seq-len back to 2nd dim
    end

    seqs = rand(1:16, 4,6)
    el = Embed(16,8)
    emb = el(seqs)
    heads = separate_heads(emb, 2)
    padd = mk_padding_mask(seqs, add_dims=true)

    dpa = dot_prod_attn(heads, heads, heads, mask=padd)

    return size(dpa[1]) == (4,4,2,6)
end


function test_mha()
    mha = MultiHeadAttn(512, 8)
    x = convert2CuArray(randn(Float32, 512, 16, 64)) 
    c,a = mha(x,x,x)

    return size(c) == (512, 16, 64) && size(a) == (16, 16, 8, 64)
end



# test transformer API:
#
    mutable struct AllYouNeed
        t::TokenTransformer
        vocab_enc
        vocab_dec
        
        AllYouNeed(n_layers, depth, heads, x_vocab, y_vocab; drop_rate=0.1) = 
            new(TokenTransformer(n_layers, depth, heads, x_vocab, y_vocab; drop_rate),
            x_vocab,
            y_vocab)
    end
    function (ayn::AllYouNeed)(x,y)   # calc loss
        
        y_in = y[1:end-1,:]       # shift y against teaching output
        y_teach = y[2:end,:]
            
        x_mask = mk_padding_mask(x)
        y_mask = mk_padding_mask(y_in)
            
        o = ayn.t(x, y_in)
            
        o_mask = (mk_padding_mask(y_teach) .== 0.0) |> Array{Float32}
        y_m = y_teach .* o_mask .|> Int   # make class ID 0 for padded positions
        loss = nll(o, y_m, average=true)  # Xentropy loss of unmasked positions only
        
        return loss
    end



function test_transformer()
    de = ["Ich liebe Julia",
          "Peter liebt Python",
          "Susi liebt sie alle",
          "Ich programmiere immer in Julia"]
    en = ["I love Julia",
          "Peter loves Python",
          "Susi loves them all",
          "I always code Julia"]
    
    de_vocab = WordTokenizer(de)
    d = de_vocab(de, add_ctls=true)
    d = pad_sequence.(d, 8)
    d = truncate_sequence.(d, 8)
    
    en_vocab = WordTokenizer(en)
    e = en_vocab(en, add_ctls=true)
    e = pad_sequence.(e, 8)
    e = truncate_sequence.(e, 8)
    
    mbs = sequence_minibatch(d, e, 2)
    x,y = first(mbs)
    
    tt =  TokenTransformer(5, 128, 4, de_vocab, en_vocab, drop_rate=0.1)
    result = tt(x,y) |> de_embed
    
    translate = AllYouNeed(5, 128, 4, de_vocab, en_vocab, drop_rate=0.1)
    translate(x,y)
    
    function tt_acc(mdl; data=nothing)
    
        tac = Float32(0.0)
        for (x,y) in data
            y_in = y[1:end-1,:]
            y_teach = y[2:end,:]
            o = mdl.t(x, y_in, embedded=false)
    
            tac += hamming_acc(o, y_teach, vocab=mdl.vocab_dec)
        end
    
        return tac / length(data)
    end
    acc = tt_acc(translate, data=mbs)

    return size(result) == (1,8,2)  && acc isa Real # seq, len, mb
end
