"""
    function tb_train!(mdl, opti, trn; epoch=1, vld=nothing, eval_size=0.2,
                      mb_loss_freq=100, eval_freq=1,
                      cp_freq=1, cp_dir="checkpoints",
                      tb_dir="tensorboard_logs", tb_name="run",
                      tb_text="""Description of tb_train!() run.""")

Train function with TensorBoard integration. TB logs are written with
the TensorBoardLogger.jl package.
The model is updated (in-place) and the trained model is returned.

### Arguments:
+ `mdl`: model; i.e. forward-function for the net
+ `opti`: Knet-stype optimiser iterator
+ `trn`: training data; iterator to provide (x,y)-tuples with
        minibatches

### Keyword arguments:
+ `epoch=1`: number of epochs to train
+ `vld=nothing`: validation data
+ `eval_size=0.2`: fraction of validation data to be used for calculating
        loss and accuracy for train and validation data during training.
+ `eval_freq=1`: frequency of evaluation; default=1 means evaluation is
        calculated after each epoch. With eval_freq=10 eveluation is
        calculated 10 times per epoch.
+ `mb_loss_freq=100`: frequency of training loss reporting. default=100
        means that 100 loss-values per epoch will be logged to TensorBoard.
        If mb_loss_freq is greater then the number of minibatches,
        loss is logged for each minibatch.
+ `cp_freq=1`: frequency of model checkpoints written to disk.
        Default is to write the model after each epoch.
+ `cp_dir="checkpoints"`: directory for checkpoints.

### TensorBoard kw-args:
TensorBoard log-directory is created from 3 parts:
`tb_dir/tb/name/<current date time>`.

+ `tb_dir="tensorboard_logs"`: root directory for tensorborad logs.
+ `tb_name="run"`: name of training run.
+ `tb_text`="""Description of tb_train!() run.""":  description
        to be included in the TensorBoard log.
"""
function tb_train!(mdl, opti, trn; epoch=1, vld=nothing, eval_size=0.1,
                  mb_loss_freq=100, eval_freq=1,
                  cp_freq=1, cp_dir="checkpoints",
                  tb_dir="./tensorboard_logs", tb_name="run",
                  tb_text="""Description of tb_train!() run.""")

    # use every n-th mb for evaluation (based on vld if defined):
    #
    n_trn = length(trn)
    n_vld = vld != nothing ? length(vld) : 1

    if vld == nothing
        n_eval = Int(ceil(n_trn * eval_size))
    else
        n_eval = Int(ceil(n_vld * eval_size))
    end
    nth_trn = Int(cld(n_trn, n_eval))
    nth_vld = Int(cld(n_vld, n_eval))

    eval_nth = Int(cld(n_trn, eval_freq))
    mb_loss_nth = Int(cld(n_trn, mb_loss_freq))

    println("Training $epoch epochs with $n_trn minibatches/epoch (and $n_vld validation mbs).")
    println("Evaluation is performed every $eval_freq minibatches (with $n_eval mbs).")

    # mk log directory:
    #
    tb_log_dir = joinpath(tb_dir, tb_name,
                    Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"))
    # checkpoints:
    #
    n_cp = Int(ceil(n_trn * cp_freq))

    # Tensorboard logger:
    #
    tbl = TensorBoardLogger.TBLogger(tb_log_dir,
                    min_level=Logging.Info)

    # Training:
    #
    mb_losses = Float32[]
    @showprogress for (i, mb_loss) in enumerate(adam(lenet, ncycle(dtrn,1)))

        push!(mb_losses, mb_loss)
        if (i % eval_nth) == 0
            calc_and_report_loss_acc(mdl, takenth(trn, nth_trn),
                    takenth(vld, nth_vld), tbl, eval_nth)
        end

        if (i % mb_loss_nth) == 0
            TensorBoardLogger.log_value(tbl,
                    "Minibatch loss (epoch = $n_trn steps)",
                    mean(mb_losses), step=i)
            mb_losses = Float32[]
        end

        if (i % cp_nth) == 0
            write_cp(mdl, i, tb_dir)
        end
    end
    return mdl
end


function write_cp(mdl, step, dir)

    fname = joinpath(dir, "checkpoints", "checkpoint_$step.jld2")
    @save fname mdl
end

# Helper to calc loss and acc with only ONE forward run:
#
function loss_and_acc(mdl, data)

    acc = nll = len = 0.0
    for (x,y) in data
        preds = mdl(x)
        len += length(y)

        acc += Knet.accuracy(preds,y, average=false)[1]
        nll += Knet.nll(preds,y, average=false)[1]
    end

    return nll/len, acc/len
end


# Helper for TensorBoardLogger:
#
function calc_and_report_loss_acc(mdl, trn, vld, tbl, step)
        loss_trn, acc_trn = loss_and_acc(mdl, trn)
        loss_vld, acc_vld = loss_and_acc(mdl, vld)
        #     println("eval at $i: loss = $loss_trn, $loss_vld; acc =  = $acc_trn, $acc_vld")

        with_logger(tbl) do
            @info "Evaluation Loss (every $step steps)" train=loss_trn valid=loss_vld log_step_increment=step
            @info "Evaluation Accuracy (every $step steps)" train=acc_trn valid=acc_vld log_step_increment=0
    end
end



"""
    function predict_top5(mdl, x; top_n=5, classes=nothing)

Run the model mdl for data in x and prints the top 5
predictions as softmax probabilities.

### Arguments:
`top_n`: print top *n* hits instead of *5*
`classes` may be a list of human readable class labels.
"""
function predict_top5(mdl, x; top_n=5, classes=nothing)

    x = first(x)
    y = mdl(x)

    if classes == nothing
        classes = repeat(["-"], maximum(top))
    end
    for (i,o) in enumerate(eachcol(y))
        o = Knet.softmax(vec(Array(o)))
        top = sortperm(vec(Array(o)), rev=true)[1:top_n]
        println("top-$top_n hits for sample $i: $top"); flush(stdout)

        @printf("%6s  %6s   %s\n", "softmax", "#", "class label")
        for t in top
            @printf(" %6.2f  %6i   \"%s\"\n", o[t], t, classes[t])
        end
        println(" ")
    end
    return y
end
