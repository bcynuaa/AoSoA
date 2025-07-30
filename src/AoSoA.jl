#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/07/28 16:28:04
  @ license: MIT
  @ declaration: EtherBenchmark: becnmark for `Ether.jl`
  @ description:
 =#

# * ==================== packages ==================== * #

using JSON
using DataFrames
using CSV
using KernelAbstractions
using ArgParse
using ThreadPinning
using OrderedCollections
pinthreads(:cores)

# * ==================== settings ==================== * #

__ARGPARSE = ArgParseSettings()
@add_arg_table __ARGPARSE begin
    "--config"
      help = "Path to the configuration file"
      arg_type = String
      default = "configs/config.json"
end

if ispath("data") == false
    mkpath("data")
end

const ARGPARSE = parse_args(__ARGPARSE)
const CONFIG = JSON.parsefile(ARGPARSE["config"])
const IT = Int32
const FT = Float32
const backend = CONFIG["backend"]
const device = CONFIG["device"]
const n_threads = CONFIG["n_threads"]
const n_neighbourhood = CONFIG["n_neighbourhood"]
const n_loops = CONFIG["n_loops"]
const n_list = CONFIG["n_list"] .|> IT
const data_path = "data/$(device).csv"
if backend == "cpu"
    const CT = Array
    const Backend = KernelAbstractions.CPU()
elseif backend == "cuda"
    using CUDA
    const CT = CUDA.CuArray
    const Backend = CUDA.CUDABackend()
elseif backend == "rocm"
    using AMDGPU
    const CT = AMDGPU.ROCArray
    const Backend = AMDGPU.ROCBackend()
elseif backend == "oneapi"
    using oneAPI
    const CT = oneAPI.oneArray
    const Backend = oneAPI.oneAPIBackend()
elseif backend == "metal"
    using Metal
    const CT = Metal.MtlArray
    const Backend = Metal.MetalBackend()
end

# * ==================== kernel function for RowStruct or ColumnStruct ==================== * #

@kernel function addByRowStruct!(neighbourhood, index, data)
    i::IT = @index(Global)
    n::IT = @inbounds neighbourhood[i]
    # * selfaction
    @inbounds data[i, 1] += data[i, 2]
    @inbounds data[i, 2] *= data[i, 3]
    @inbounds data[i, 3] += data[i, 4]
    @inbounds data[i, 4] *= data[i, 5]
    @inbounds data[i, 5] += data[i, 96]
    @inbounds data[i, 96] *= data[i, 97]
    @inbounds data[i, 97] += data[i, 98]
    @inbounds data[i, 98] *= data[i, 99]
    @inbounds data[i, 99] += data[i, 1]
    # * interaction
    count::IT = 0
    while count < n
        j = @inbounds index[i, count + 1]
        @inbounds data[i, 1] += data[j, 2]
        @inbounds data[i, 2] *= data[j, 3]
        @inbounds data[i, 3] += data[j, 4]
        @inbounds data[i, 4] *= data[j, 5]
        @inbounds data[i, 5] += data[j, 96]
        @inbounds data[i, 96] *= data[j, 97]
        @inbounds data[i, 97] += data[j, 98]
        @inbounds data[i, 98] *= data[j, 99]
        @inbounds data[i, 99] += data[j, 1]
        # * count ++
        count += 1
    end
end

@kernel function addByColumnStruct!(neighbourhood, index, data)
    i::IT = @index(Global)
    n::IT = @inbounds neighbourhood[i]
    # * selfaction
    @inbounds data[1, i] += data[2, i]
    @inbounds data[2, i] *= data[3, i]
    @inbounds data[3, i] += data[4, i]
    @inbounds data[4, i] *= data[5, i]
    @inbounds data[5, i] += data[96, i]
    @inbounds data[96, i] *= data[97, i]
    @inbounds data[97, i] += data[98, i]
    @inbounds data[98, i] *= data[99, i]
    @inbounds data[99, i] += data[1, i]
    # * interaction
    count::IT = 0
    while count < n
        j = @inbounds index[count + 1, i]
        @inbounds data[1, i] += data[2, j]
        @inbounds data[2, i] *= data[3, j]
        @inbounds data[3, i] += data[4, j]
        @inbounds data[4, i] *= data[5, j]
        @inbounds data[5, i] += data[96, j]
        @inbounds data[96, i] *= data[97, j]
        @inbounds data[97, i] += data[98, j]
        @inbounds data[98, i] *= data[99, j]
        @inbounds data[99, i] += data[1, j]
        # * count ++
        count += 1
    end
end

# * ==================== array prepare ==================== * #

@inbounds function prepareArray(n, transpose = false)
    interval = ceil(n / n_neighbourhood) |> IT
    if transpose == false
        index = zeros(IT, n, n_neighbourhood)
        data = randn(FT, n, 100)
        Threads.@threads for i in 1:n
            for j in 1:n_neighbourhood
                id = i + (j - 1) * interval
                id = mod1(id, n)
                @inbounds index[i, j] = id
            end
        end
    else
        index = zeros(IT, n_neighbourhood, n)
        data = randn(FT, 100, n)
        Threads.@threads for i in 1:n
            for j in 1:n_neighbourhood
                id = i + (j - 1) * interval
                id = mod1(id, n)
                @inbounds index[j, i] = id
            end
        end
    end
    return index, data
end

@inbounds function parseTimedReport(report)::OrderedDict{String, Float64}
    timed_report = OrderedDict{String, Float64}()
    timed_report["time"] = report.time
    timed_report["gctime"] = report.gctime
    timed_report["compile_time"] = report.compile_time
    timed_report["recompile_time"] = report.recompile_time
    return timed_report
end

@inbounds function singleRun(n, transpose)::Tuple
    index, data = prepareArray(n, transpose)
    index = CT(index)
    data = CT(data)
    neighbourhood = zeros(IT, n)
    fill!(neighbourhood, n_neighbourhood)
    neighbourhood = CT(neighbourhood)
    # * cause compilation
    if transpose == false
        kernel! = addByRowStruct!(Backend, n_threads, (Int64(n),))
    else
        kernel! = addByColumnStruct!(Backend, n_threads, (Int64(n),))
    end
    kernel!(neighbourhood, index, data, ndrange = (n,))
    # * run kernel
    report = @timed begin
        for _ in 1:n_loops
            kernel!(neighbourhood, index, data, ndrange = (n,))
            KernelAbstractions.synchronize(Backend)
        end
    end
    index = nothing
    data = nothing
    neighbourhood = nothing
    GC.gc()
    timed_report = parseTimedReport(report)
    major = transpose == false ? "row" : "column"
    return Tuple(vcat(
        [device, backend, major, n],
        Float64.(collect(values(timed_report))),
    ))
end

@inline function main()
    df = DataFrame(device=String[], backend=String[], major=String[], n = Int[], time=Float64[], gctime=Float64[], compile_time=Float64[], recompile_time=Float64[])
    for n in n_list
        for transpose in (false, true)
            println("Running for n = $n, transpose = $transpose")
            result = singleRun(n, transpose)
            push!(df, result)
            CSV.write(data_path, df, header = true, delim=",")
        end
    end
end

main()
