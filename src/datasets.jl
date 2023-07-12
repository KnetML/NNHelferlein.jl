# funs for handling the example data, not handled by default GIT
#

const ZENODO_URL = "https://zenodo.org"

# no y/n dialog if a Download is requested
#
ENV["DATADEPS_ALWAYS_ACCEPT"] = true


# PFAM protein families database:
# Curated sequences and families
#
#
#
const ZENODO_DATA_PFAM = "8138939"
const URL_DATA_PFAM = "$ZENODO_URL/record/$ZENODO_DATA_PFAM/files"
const PFAM_DIR = "pfam"
const PFAM_RAW_FILE = "pfam_46872x62.csv"
const PFAM_FAMILIES_FILE = "families.csv"
const PFAM_AA_FILE = "aminoacids.csv"

const PFAM_TRAIN_XY_FILE = "pfam-trn-xy.csv"
const PFAM_TRAIN_LB_FILE = "pfam-trn-labels.csv"
const PFAM_TEST_XY_FILE = "pfam-tst-xy.csv"
const PFAM_TEST_LB_FILE = "pfam-tst-labels.csv"
const PFAM_BAL_TRAIN_XY_FILE = "pfam-balanced-trn-xy.csv"
const PFAM_BAL_TRAIN_LB_FILE = "pfam-balanced-trn-labels.csv"

"""
    function dataset_pfam(records; force=false)

Retrieve the curated PFAM protein families database from Zenodo including
46872 sequences from 62 families. Sequences are between 100 and 1000 amino acids long
and families have between 100 and 200 memebers.
Training and test data are padded to a length of 1000 amino acids with the padding token of
the amino acid tokenizer (26).

More information about the data set can be found at 
<https://zenodo.org/record/8138939>, including PDB sequence IDs for each data table.

## Available records:

+ `:raw`: dataframe with all (46872) rows of data and the columns *ID* (PDB-ID), 
            *family* (family name) and *sequence* (amino acid sequence)
+ `:families`: list of all family names as dataframe with the  columns 
            *class* (cnumeric class ID 1-62), *family* (family name) and
            and *count* (number of family members in the dataset)
+ `:aminoacids`: list of amino acid tokes as dataframe with the columns
            *Token* (aa token 1-26), *One-Letter* (one-letter code of the amino acid),
            and *Amino acid* (full name of the amino acid)
+ `:train`: dataframe with 42187 rows of training data and labels
            with the class ID as first column and the 
            amino acid tokens as columns 2-1001 (padded to 1000 amino acids)
+ `:test`: dataframe with 4687 rows of test data in the same format as the training data
+ `:balanced_train`: dataframe with 111601 rows of balanced training data in the same format 
            as the training data. The data is balanced by sampling 1800 sequences from each family.
+ `:balanced_test`: dataframe with 12401 rows of balanced test data in the same format as the training data.
"""
function dataset_pfam(records; force=false)

    if records == :raw
        return download_pfam(PFAM_RAW_FILE, "raw PFAM family dataset", force=force)
    elseif records == :families
        return download_pfam(PFAM_FAMILIES_FILE, "protein families", force=force)
    elseif records == :amoinoacids
        return download_pfam(PFAM_AA_FILE, "amino acid tokens", force=force)
    elseif records == :train
        return download_pfam(PFAM_TRAIN_XY_FILE, "x/y training data", force=force)
    elseif records == :test
        return download_pfam(PFAM_TEST_XY_FILE, "x/y test data", force=force)
    elseif records == :balanced_train
        return download_pfam(PFAM_BAL_TRAIN_XY_FILE, "x/y balanced training data", force=force)
    elseif records == :balanced_test
        return download_pfam(PFAM_BAL_TEST_FILE, "x/y balanced test data", force=force)
    else
        println("Unable to download unknown records $records")
    end
end

function download_pfam(fname, msg; force=true)

    dir=joinpath(NNHelferlein.DATA_DIR, PFAM_DIR)
    if !isdir(dir)
        mkpath(dir)
    end

    local_file = joinpath(dir, fname)
    url = "$URL_DATA_PFAM/$fname?download=1"

    if !isfile(local_file) || force
        println("  downloading  $msg"); flush(stdout)
        Downloads.download(url, local_file)
    else
        println("  skiping download for $msg (use force=true to overwrite local copy)")
    end

    return CSV.read(local_file, DataFrame)
end




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
numeric array of eltype `Float32`. 
The values are scaled to be between 0 and 1. 
The labels are returned as a vector of `Int8`.

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

    xtrn,ytrn,xtst,ytst = dataset_from_mldatasets(
                            MNIST, MNIST_DIR, force=force)
    
    ytrn[ytrn.==0] .= 10
    ytrn = Int8.(ytrn)
    
    ytst[ytst.==0] .= 10
    ytst = Int8.(ytst)

    return xtrn, ytrn, xtst, ytst
end
    

const FASHION_MNIST_DIR = "fashion_mnist"

"""
    function dataset_fashion_mnist(; force=false)

Download Zalando's Fashion-MNIST datset with help of `MLDatasets.jl` 
from https://github.com/zalandoresearch/fashion-mnist.

4 arrays `xtrn, ytrn, xtst, ytst` are returned in the 
same structure as the original MNIST dataset.


The data is stored in the *Helferlein* data directory and only downloaded
the files are not already saved.

Authors: Han Xiao, Kashif Rasul, Roland Vollgraf

### Arguments:
+ `force=false`: if `true`, the dataset download will be forced.
"""
function dataset_fashion_mnist(; force=false)

    xtrn,ytrn,xtst,ytst = dataset_from_mldatasets(
                            FashionMNIST, FASHION_MNIST_DIR, force=force)
    
    ytrn[ytrn.==0] .= 10
    ytrn = Int8.(ytrn)
    
    ytst[ytst.==0] .= 10
    ytst = Int8.(ytst)

    return xtrn, ytrn, xtst, ytst
end









function dataset_from_mldatasets(DataSet, dir; Tx=Float32, force=false)

    data_dir = joinpath(NNHelferlein.DATA_DIR, dir)

    if force && isdir(data_dir)
        rm(data_dir, force=true, recursive=true)
    end

    if !isdir(data_dir)
        println("Downloading dataset ...")
    else
        println("Dataset is already downloaded at $data_dir.")
    end

    trn = DataSet(; Tx=Float32, split=:train, dir=data_dir)
    xtrn, ytrn = trn.features, trn.targets
    
    tst = DataSet(; Tx=Float32, split=:test, dir=data_dir)
    xtst, ytst = tst.features, tst.targets
    
    return xtrn, ytrn, xtst, ytst
end



# IRIS data:
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



