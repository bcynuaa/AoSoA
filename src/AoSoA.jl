#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/07/28 16:28:04
  @ license: MIT
  @ declaration: EtherBenchmark: becnmark for `Ether.jl`
  @ description:
 =#

using JSON
using DataFrames
using CSV
using KernelAbstractions
using ArgParse

__ARGPARSE = ArgParseSettings()
@add_arg_table __ARGPARSE begin
    "--config"
      help = "Path to the configuration file"
      arg_type = String
      default = "configs/config.json"
end

const ARGPARSE = parse_args(__ARGPARSE)
println(ARGPARSE)

const IT = Int32
const FT = Float32

@kernel function addByRowStruct!(neighbourhood, index, data)
    i::IT = @index(Global)
    n::IT = @inbounds neighbourhood[i]
    count::IT = 0
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
    while count < n
        j = @inbounds index[i, count]
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
    count::IT = 0
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
    while count < n
        j = @inbounds index[i, count]
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