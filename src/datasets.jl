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
const MIT_NSR_RECORDS = ["16265", "16272", "16273", "16420",
    "16483", "16539", "16773", "16786",
    "16795", "17052", "17453", "18177",
    "18184", "19088", "19090", "19093",
    "19140", "19830"]

const MIT_NSR_DIR = "MIT-normal_sinus_rhythm"


function download_mit_nsr(records; force=false, dir=joinpath(NNHelferlein.DATA_DIR, MIT_NSR_DIR))

    println("Downloading MIT-Normal Sinus Rhythm Database from Zenodo ...")

    if !isdir(dir)
        mkpath(dir)
    end

    for (i, record) in enumerate(records)

        local_file = joinpath(dir, record)
        url = "$URL_DATA_NSR_BEATS/$record?download=1"

        if !isfile(local_file) || force
            println("  downloading $i of $(length(records)): $record"); flush(stdout)
            Downloads.download(url, local_file)
        else
            println("  skiping download for record $record (use force=true to overwrite local copy)")
        end
    end
end

"""
    function dataset_mit_nsr(records=nothing; force=false)

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
+ `records`: list of records names to be downloaded.

### Examples:

```juliaREPL
nsr_16265 = dataset_mit_nsr("16265")
nsr_16265 = dataset_mit_nsr(["16265", "19830"])
nsr_all = dataset_mit_nsr()
```
"""
function dataset_mit_nsr(records=nothing; force=false)

    function read_ecg(record)
        fname = joinpath(NNHelferlein.DATA_DIR, MIT_NSR_DIR, record)
        x = CSV.File(fname, types=[Float32, Float32, Float32, 
                Float32, Float32] ) |> DataFrames.DataFrame
        x.time = collect(1:DataFrames.nrow(x)) ./ 128
        return x
    end

    if records == nothing
        records = MIT_NSR_RECORDS
    elseif records isa AbstractString
        records = [records]
    end

    records = records .* ".ecg.gz"

    
    dataframes = nothing
    try
        download_mit_nsr(records, force=force)
        dataframes = [read_ecg(record) for record in records]
    catch
        println("Error downloading dataset from Zenodo - please try again later!")
        dataframes = nothing
    end
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
4 arrays `xtrn, ytrn, xtst, ytst` are returned. 

`xtrn` and `xtst` will be the images as a multi-dimensional
array, and `ytrn` and `ytst` the corresponding labels as integers.

The image(s) is/are returned in the horizontal-major memory layout as a single
numeric array of eltype T. If T <: Integer, then all values will be within 0 and
255, otherwise the values are scaled to be between 0 and 1. The integer values
of the labels correspond 1-to-1 the digit that they represent.

In the 
teaching input (i.e. `y`) the digit `0` is encoded as `10`.

The data is stored in the *Helferlein* data directory and only downloaded
the files are not already saved.

Ref.:  Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner.
"Gradient-based learning applied to document recognition."
*Proceedings of the IEEE,* 86(11):2278-2324, November 1998      
<http://yann.lecun.com/exdb/mnist/>.


### Arguments:
+ `force=false`: if `true`, the dataset download will be forced.
"""
function dataset_mnist(; force=false)

    mnist_dir = joinpath(NNHelferlein.DATA_DIR, MNIST_DIR)

    if force && isdir(mnist_dir)
        rm(mnist_dir, force=true, recursive=true)
    end

    # pre-download:
    #
    if !isdir(mnist_dir)
        MLDatasets.MNIST.download(mnist_dir, i_accept_the_terms_of_use=true)
    end

    # read:
    #
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




# TODO datasets fashion mnist und weitere einfache