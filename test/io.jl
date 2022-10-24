lenet = Classifier(Conv(5,5,1,20),       
                Pool(),
                Conv(5,5,20,50),
                Pool(),
                Flat(),
                Dense(800,512), 
                Dense(512,10, actf=identity)
        )

function test_copy()

    c1 = copy_network(lenet, to=:gpu)
    c2 = copy_network(lenet, to=:cpu)

    return lenet.loss == c1.loss &&
           lenet.loss == c2.loss
end


function test_save()

    save_network("save_lenet", lenet)
    c1 = load_network("save_lenet", to=:gpu)
    c2 = load_network("save_lenet", to=:cpu)

    return lenet.loss == c1.loss &&
           lenet.loss == c2.loss
end

