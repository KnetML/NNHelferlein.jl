# Data loader and preprocessing funs for dataframes:
#
# (c) A. Dominik, 2020

"""
    dataframe_read(fname)

Read a data table from an CSV-file with one sample
per row and return a DataFrame with the data.
(ODS-support is removed because of frequent PyCall compatibility issues
of the OdsIO package).
"""
function dataframe_read(fName)

    if occursin(r".*\.ods$", fname)
        println("Reading ODS-files is no longer supported!")
        return nothing
        # return readODS(fname)
    elseif occursin(r".*\.csv$", fname)
        return readCSV(fname)
    else
        println("Error: unknown file format!")
        println("Please support csv or ods file")
        return nothing
    end
end

# function readODS(fname)
#
#     printf("Reading data from ODS: $fname")
#     return OdsIO.ods_read(fname, retType="DataFrame")
# end

function readCSV(fname)

    printf("Reading data from CSV: $fname")
    return DataFrames.DataFrame(CSV.File(fname, header=true))
end



"""
    dataframe_minibatches(data::DataFrames.DataFrame; size=256, ignore=[], teaching=:y, regression=true)

Make Knet-conform minibatches from a dataframe
with one sample per row.

+ `ignore`: defines a list of column names to be ignored
+ `teaching`: defines the column with teaching input. Default is ":y"
+ `regression`: if `true`, the teaching input is interpreted as
                scalar value (converted to Float32); otherwise
                teaching input is used as class labels (converted
                to UInt8).
"""
function dataframe_minibatches(data; size=256, ignore=[], teaching=:y, regression=true)

    if teaching == nothing
        x = Matrix(transpose(Array{Float32}(select(data, Not(ignore)))))
        return Knet.minibatch(x, size)
    else
        push!(ignore, teaching)
        x = Matrix(transpose(Array{Float32}(select(data, Not(ignore)))))
        if regression
            y = Matrix(transpose(Array{Float32}(data[teaching])))
        else
            y = Matrix(transpose(Array{UInt8}(data[teaching]))) .+ UInt8(1)
        end
        return Knet.minibatch(x, y, size, partial=true)
    end
end


"""
    function dataframe_split(df::DataFrames.DataFrame; teaching=:y, fr=0.2, balanced=true)

Split data, organised row-wise in a DataFrame into train and valid sets.

### Arguments:
+ `df`: data
+ `teaching`: name or index of column with teaching input (y)
+ `fr`: fraction of data to be used for validation
+ `balanced`: if `true`, result datasets will be balanced by oversampling.
              Returned datasets will be bigger as expected
              but include the same numbers of samples for each class.
"""
function dataframe_split(df::DataFrames.DataFrame; teaching=:y,
                         fr=0.2, balanced=false)

    ((trn,ytrn), (vld,yvld)) = do_split(df, df[teaching], at=fr)

    if balanced
        (trn,ytrn) = do_balance(trn, ytrn)
        (vld,yvld) = do_balance(vld, yvld)
    end

    return trn, vld
end