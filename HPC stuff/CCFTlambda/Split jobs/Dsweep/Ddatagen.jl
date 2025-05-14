using CSV, DataFrames, DelimitedFiles

Ds = 10:5:100
Ds = Vector(Ds)
writedlm("HPC stuff/CCFTlambda/Split jobs/Dsweep/Dvalues.csv", Ds, ',')