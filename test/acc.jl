# funs for test of acc-funs:
#

function test_peak_finder()

    t = [0, 0,   0,   0, 0,   1,   0,   0,   0, 0, 0, 1]
    y = [0, 0.1, 0.1, 0, 0.8, 0,   0,   0, 0.9, 0, 0, 0]

    f1 = peak_finder_acc(y, t, verbose=3)  # != 2/3
    gm = peak_finder_acc(y, t, verbose=1, ret=:g_mean)  # ≈ 0.7071
    iou = peak_finder_acc(y, t, verbose=1, ret=:iou)  # 
    return f1 ≈ 2/3 && gm ≈ 0.7071067811865 && iou ≈ 0.5
end



function test_peak_finder_acc()

    t = [0, 0,   0,   0, 0,   1,   0,   0,   0, 0, 0, 1]
    y = [0, 0.1, 0.1, 0, 0.8, 0,   0,   0, 0.9, 0, 0, 0]
    d = [(y,t), (y,t)]

    mdl(x) = x
    f1 = peak_finder_acc(mdl, data=d)
    return f1 ≈ 2/3 
end



function test_hamming()

        p = [0  0  0  0  0
             4  4  3  4  5
             2  2  2  2  5
             1  3  1  3  3
             0  0  0  0  0]
        t = [0  0  0  0  0
             4  4  5  3  4
             3  1  2  4  4
             2  5  5  4  2
             0  0  0  0  0]

        acc = hamming_dist(p,t)
        hd = hamming_acc(p,t)

       return isapprox(acc, 0.6, atol=0.05) && isapprox(hd, 0.77, atol=0.05)
end


function test_hamming_acc()

        p = [0  0  0  0  0
             4  4  3  4  5
             2  2  2  2  5
             1  3  1  3  3
             0  0  0  0  0]
        t = [0  0  0  0  0
             4  4  5  3  4
             3  1  2  4  4
             2  5  5  4  2
             0  0  0  0  0]

        d = [(p,t), (p,t)]
        mdl(x) = x

        acc = hamming_acc(mdl, data=d)
        
       return isapprox(acc, 0.76, atol=0.1)
end



function test_hamming_vocab()
    tok = WordTokenizer(["I love Julia",
                         "Peter loves Python",
                         "We all marvel Geoff"])
    l = tok(["I love Julia", "Peter loves Python", "We all marvel Geoff"],
            add_ctls=true)

     h1 = hamming_dist([1, 7, 9, 8, 2], [1, 5, 9, 8, 2], vocab=tok)  # =! 1

     return h1 == 1
end


function test_hamming_length()
    tok = WordTokenizer(["I love Julia",
                         "Peter loves Python",
                         "We all marvel Geoff"])
    l = tok(["I love Julia", "Peter loves Python", "We all marvel Geoff"],
            add_ctls=true)
     p = [1, 7, 9, 8, 2]
     t = [1, 7, 9, 2]
     h1 = hamming_dist(p, t, vocab=tok)  # =! 1
     h2 = hamming_dist(t, p, vocab=tok)  # =! 2

     return h1 == 1 && h2 == 2
end



function test_confusion_matrix()
     
     # returns 1 for the first half of x and 2 for the rest.
     #
     function mdl(x)
          half_1 = length(x) ÷ 2
          half_2 = length(x) - half_1

          return hcat(repeat([1;0], 1,half_1), repeat([0;1], 1, half_2))
     end

     mb = [(collect(1:10), [1 1 1 1 1 2 2 2 2 2])]
     c = confusion_matrix(mdl, data=mb, pretty_print=false, accuracy=false)
     return c ≈ [5 0; 0 5]
end


function test_mb_eval()

     dat = (([1,2,3,4], [1,1,1,1]), ([1,2], [1,1]))
     mdl(x) = ifgpu(x) .* 2

     sq_err = squared_error_acc(mdl, data=dat)
     ab_err = abs_error_acc(mdl, data=dat)

     return isapprox(sq_err, 15.6667, atol=0.001) &&
            isapprox(ab_err,  3.3333, atol=0.001)
end

function test_focal_loss()

     y = [0, 1, 2]
     s = [0.8 0.1 0.3
          0.1 0.9 0.0
          0.0 0.3 1.0]
     fnll = focal_nll(s, y)  # 0.9711265498008855

     y = [0, 1, 0]
     s = [0.2, 1.0, 0.1]
     fbce = focal_bce(s, y) # 0.11745655310473864

     return isapprox(fnll, 0.97112, atol=0.001) &&
            isapprox(fbce, 0.11745, atol=0.001)
end
     
