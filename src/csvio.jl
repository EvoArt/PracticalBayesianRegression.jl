# =============================================================================
# A small CSV reader and draws writer for model binaries (binary.jl).
#
# Why not CSV.jl: it builds column types at run time (that is its job), which
# is exactly what `juliac --trim` cannot compile. A model binary knows what
# each column must be from its recipe, so the reader only needs strings: every
# cell is kept as written (level names then match what R / brms would print:
# a group column "1, 2, 3" gives levels "1", "2", "3", not "1.0") plus its
# Float64 parse. RFC 4180 quoting; `NA` and the empty cell are missing (R's
# read.csv defaults).
# =============================================================================

"""
    DataTable

A CSV file read by [`read_csv`](@ref): its column names, and for each column
the cells as written, their numeric values (`NaN` where a cell is not a
number) and which cells are missing (`NA` or empty).
"""
struct DataTable
    names::Vector{String}
    raw::Vector{Vector{String}}
    num::Vector{Vector{Float64}}
    miss::Vector{BitVector}
end

_nrows(t::DataTable) = isempty(t.raw) ? 0 : length(t.raw[1])

function _colindex(t::DataTable, name::String)
    k = findfirst(==(name), t.names)
    k === nothing && throw(ArgumentError("the data has no column `$name` (columns: $(join(t.names, ", ")))"))
    return k
end

_csv_missing(s::String) = isempty(s) || s == "NA"

# TRUE / FALSE (R's logicals) count as numbers for numeric atoms: a bernoulli
# response written by R's write.csv is TRUE/FALSE
function _csv_num(s::String)
    s == "TRUE" && return 1.0
    s == "FALSE" && return 0.0
    v = tryparse(Float64, s)
    return v === nothing ? NaN : v
end

# split one record into fields; `i` is the position after the record's end
function _csv_record!(fields::Vector{String}, text::String, i::Int, delim::Char)
    empty!(fields)
    n = ncodeunits(text)
    buf = IOBuffer()
    inq = false
    while i <= n
        c = text[i]
        if inq
            if c == '"'
                j = nextind(text, i)
                if j <= n && text[j] == '"'          # "" inside quotes is a literal quote
                    write(buf, '"'); i = nextind(text, j); continue
                end
                inq = false
            else
                write(buf, c)
            end
        elseif c == '"'
            inq = true
        elseif c == delim
            push!(fields, String(take!(buf)))
        elseif c == '\n' || c == '\r'
            j = nextind(text, i)
            c == '\r' && j <= n && text[j] == '\n' && (j = nextind(text, j))
            push!(fields, String(take!(buf)))
            return j
        else
            write(buf, c)
        end
        i = nextind(text, i)
    end
    push!(fields, String(take!(buf)))
    return n + 1
end

"""
    read_csv(path; delim = ',') -> DataTable

Read a CSV file with a header row, as a model binary does. Quoted fields
follow RFC 4180; `NA` and empty cells are missing, as in R's `read.csv`.
"""
read_csv(path::AbstractString; delim::Char=',') = parse_csv(read(String(path), String); delim=delim)

function parse_csv(text::String; delim::Char=',')
    n = ncodeunits(text)
    fields = String[]
    i = 1
    # a UTF-8 byte-order mark (Excel writes one)
    startswith(text, "﻿") && (i = nextind(text, 1))
    i = _csv_record!(fields, text, i, delim)
    names = [strip(f) |> String for f in fields]
    any(isempty, names) && throw(ArgumentError("the CSV header has an empty column name"))
    K = length(names)
    raw = [String[] for _ in 1:K]
    line = 1
    while i <= n
        i = _csv_record!(fields, text, i, delim)
        line += 1
        length(fields) == 1 && isempty(fields[1]) && continue      # blank line
        length(fields) == K ||
            throw(ArgumentError("CSV line $line has $(length(fields)) fields; the header has $K"))
        for k in 1:K
            push!(raw[k], fields[k])
        end
    end
    num = [map(_csv_num, r) for r in raw]
    miss = [BitVector(map(_csv_missing, r)) for r in raw]
    return DataTable(names, raw, num, miss)
end

"""
    write_draws_csv(io, names, draws[, statnames, stats])

`draws` is (iterations x chains x variables) on the constrained scale; one row
per draw, chains stacked, with `chain` and `iteration` columns first, then the
sampler statistics `stats` (iterations x chains x statnames) when given.
"""
function write_draws_csv(io::IO, names::Vector{String}, draws::Array{Float64,3},
                         statnames::Vector{String}=String[],
                         stats::Array{Float64,3}=Array{Float64,3}(undef, size(draws, 1), size(draws, 2), 0))
    print(io, "chain,iteration")
    for nm in statnames
        print(io, ',', nm)
    end
    for nm in names
        print(io, ',')
        _csv_field(io, nm)
    end
    print(io, '\n')
    niter, nch, nv = size(draws)
    ns = size(stats, 3)
    for c in 1:nch, it in 1:niter
        print(io, c, ',', it)
        for k in 1:ns
            print(io, ',', stats[it, c, k])
        end
        for v in 1:nv
            print(io, ',', draws[it, c, v])
        end
        print(io, '\n')
    end
    return nothing
end

function _csv_field(io::IO, s::String)
    if occursin(',', s) || occursin('"', s)
        print(io, '"', replace(s, "\"" => "\"\""), '"')
    else
        print(io, s)
    end
end
