# funs for handling the example data, not handled by default GIT
#

const ZENODO_URL = "https://zenodo.org"

# ECG data: MIT Normal Sinus Rhythm database:
#
#
#

# download ECG data:
#
const ZENODO_DATA_NSR_BEATS = "6526342"   # Zenodo identifier

const URL_DATA_NSR_BEATS = "$ZENODO_URL/record/$ZENODO_DATA_NSR_BEATS/files"
const RECORDS = ["16265.ecg.gz", "16272.ecg.gz", "16273.ecg.gz", "16420.ecg.gz",
    "16483.ecg.gz", "16539.ecg.gz", "16773.ecg.gz", "16786.ecg.gz",
    "16795.ecg.gz", "17052.ecg.gz", "17453.ecg.gz", "18177.ecg.gz",
    "18184.ecg.gz", "19088.ecg.gz", "19090.ecg.gz", "19093.ecg.gz",
    "19140.ecg.gz", "19830.ecg.gz"]

const MIT_NSR_DIR = "MIT-normal_sinus_rhythm"


function download_mit_nsr(; force=false, dir=joinpath(NNHelferlein.DATA_DIR, MIT_NSR_DIR))

    println("Downloading MIT-Normal Sinus Rhythm Database from Zenodo ...")

    if !isdir(dir)
        mkpath(dir)
    end

    for (i, record) in enumerate(RECORDS)

        local_file = joinpath(dir, record)
        url = "$URL_DATA_NSR_BEATS/$record?download=1"

        if !isfile(local_file) || force
            println("  downloading $i of $(length(RECORDS)): $record"); flush(stdout)
            download(url, local_file)
        else
            println("  skiping download for record $record (use force=true to overwrite local copy)")
        end
    end
end

"""
    function dataset_mit_nsr(; force=false)

Retrieve the Physionet ECG data set: "MIT-BIH Normal Sinus Rhythm Database".
If necessary the data is downloaded from Zenodo (and stored in the *NNHelferlein*
data directory, 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6526342.svg)](https://doi.org/10.5281/zenodo.6526342)).

All 18 recordings are returned as a list of DataFrames.

ECGs from the MIT-NSR database with some modifications to make them more 
suitable as playground data set for machine learning.

* all 18 ECGs are trimmed to approx. 50000 heart beats from a region 
  without recording errors
* scaled to a range -1 to 1 (non-linear/tanh)
* heart beats annotation as time series with 
  value 1.0 at the point of the annotated beat and 0.0 for all other times
* additional heart beat column smoothed by applying a gaussian filter
* provided as csv with columns "time in sec", "channel 1", "channel 2", 
  "beat" and  "smooth".

### Arguments:
+ `force=false`: if `true` the download will be forced and local data will be 
        overwitten.
"""
function dataset_mit_nsr(; force=false)

    function read_ecg(record)
        fname = joinpath(NNHelferlein.DATA_DIR, MIT_NSR_DIR, record)
        x = CSV.File(fname, types=[Float32, Float32, Float32, 
                Float32, Float32] ) |> DataFrames.DataFrame
        x.time = collect(1:DataFrames.nrow(x)) ./ 128
        return x
    end

    download_mit_nsr(force=force)
    dataframes = [read_ecg(record) for record in RECORDS]
    return dataframes
end





# MNIST data:
#
#
#

const MNIST_DIR = "mnist"

"""
    function dataset_mnist(; force=false)

Download the MNIST dataset with help of `MLDatasets.jl` from 
Yann LeCun's official website.
4 arrays `xtrn, ytrn, xtst, ytst` are returned. In the 
teaching input (i.e. `y`) the digit `0` is encoded as `10`.

The data is stored in the *Helferlein* data directory and only downloaded
the files are not already saved.

Ref.:  Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner.
"Gradient-based learning applied to document recognition."
*Proceedings of the IEEE,* 86(11):2278-2324, November 1998      
<http://yann.lecun.com/exdb/mnist/>.


### Arguments:
+ `force=false`: if `false`, the dataset download will be forced.
"""
function dataset_mnist(; force=false)

    mnist_dir = joinpath(NNHelferlein.DATA_DIR, MNIST_DIR)

    if force && isdir(mnist_dir)
        rm(mnist_dir, force=true, recursive=true)
    end

    xtrn,ytrn = MNIST.traindata(Float32, dir=mnist_dir)
    ytrn[ytrn.==0] .= 10
    
    xtst,ytst = MNIST.testdata(Float32, dir=mnist_dir)
    ytst[ytst.==0] .= 10
    
    return xtrn, ytrn, xtst, ytst
end



# IRIS data:
#
#
#
const IRIS_DIR = "iris"
const IRIS_CSV = "iris150.csv"

"""
    function dataset_iris()

Return Fisher's *iris* dataset of 150 records as dataframe.

Ref: Fisher,R.A. 
     "The use of multiple measurements in taxonomic problems" 
     *Annual Eugenics*, 7, Part II, 179-188 (1936); 
     also in "Contributions to Mathematical Statistics" (John Wiley, NY, 1950).     
     <https://archive.ics.uci.edu/ml/datasets/Iris>
"""
function dataset_iris()

    return dataframe_read(joinpath(NNHelferlein.DATA_DIR, IRIS_DIR, IRIS_CSV))
end