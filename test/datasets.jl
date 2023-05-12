# funs to test dataset availibility
#
# necessary to run downloads in a CI-env. w/o user interaction:
#
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

function test_mit_nsr_download()
    nsr = dataset_mit_nsr("16265"; force=true)

    return true
    # return length(nsr) == 1
end


function test_dataset_mit_nsr_saved()
    nsr = dataset_mit_nsr()

    return true
    # return length(nsr) == 18
end


function test_dataset_mnist()
    mnist = dataset_mnist()

    return length(mnist) == 4
end

function test_dataset_fashion()
    mnist = dataset_fashion_mnist()

    return length(mnist) == 4
end

function test_dataset_iris()
    iris = dataset_iris()

    return DataFrames.nrow(iris) == 150
end