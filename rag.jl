using Base.Threads
using StaticArrays
using JSON
using HTTP

struct RAGServer
  ids::Vector{Int64}
  data::Matrix{UInt64}
end

function start_server(rag::RAGServer; port::Int=8003)
  router = HTTP.Router()

  HTTP.register!(router, "POST", "/find_matches", function (req)
    body = JSON.parse(String(req.body))
    query = UInt8.(body["query"])
    k = get(body, "k", 20)

    if length(query) != 64
      return HTTP.Response(400, "Query must be an array of 64 UInt8 elements")
    end

    query_uint64 = SVector{8,UInt64}(reinterpret(UInt64, query))
    t1 = time()
    results = k_closest_parallel(rag.data, query_uint64, k)
    t2 = time()
    println("Time taken for query $(t2 - t1)")

    response = [
      Dict("id" => rag.ids[result.second], "distance" => result.first)
      for result in results
    ]

    return HTTP.Response(200, JSON.json(response))
  end)

  HTTP.serve(router, "0.0.0.0", port)
end

function load_data_from_bin()
  open("bindata/data.bin", "r") do file
    num_columns = read(file, UInt64)
    num_rows = read(file, UInt64)
    data = Matrix{UInt64}(undef, num_rows, num_columns)
    read!(file, data)

    ids = Vector{Int64}(undef, num_columns)
    open("bindata/ids.bin", "r") do id_file
      read!(id_file, ids)
    end

    return ids, data
  end
end


@inline function hamming_distance(s1::AbstractString, s2::AbstractString)::Int
  s = 0
  for (c1, c2) in zip(s1, s2)
    if c1 != c2
      s += 1
    end
  end
  s
end

@inline function hamming_distance(x1::T, x2::T)::Int where {T<:Integer}
  return Int(count_ones(x1 ⊻ x2))
end

@inline function hamming_distance1(
  x1::AbstractArray{T},
  x2::AbstractArray{T},
)::Int where {T<:Integer}
  s = 0
  for i in eachindex(x1, x2)
    s += hamming_distance(x1[i], x2[i])
  end
  s
end

@inline function hamming_distance(
  x1::AbstractArray{T},
  x2::AbstractArray{T},
)::Int where {T<:Integer}
  s = 0
  @inbounds @simd for i in eachindex(x1, x2)
    s += hamming_distance(x1[i], x2[i])
  end
  s
end


mutable struct MaxHeap
  const data::Vector{Pair{Int,Int}}
  current_idx::Int # add pairs until current_idx > length(data)
  const k::Int

  function MaxHeap(k::Int)
    new(fill((typemax(Int) => -1), k), 1, k)
  end
end

function insert!(heap::MaxHeap, value::Pair{Int,Int})
  if heap.current_idx <= heap.k
    heap.data[heap.current_idx] = value
    heap.current_idx += 1
    if heap.current_idx > heap.k
      makeheap!(heap)
    end
  elseif value.first < heap.data[1].first
    heap.data[1] = value
    heapify!(heap, 1)
  end
end

function makeheap!(heap::MaxHeap)
  for i = div(heap.k, 2):-1:1
    heapify!(heap, i)
  end
end

function heapify!(heap::MaxHeap, i::Int)
  left = 2 * i
  right = 2 * i + 1
  largest = i

  if left <= length(heap.data) && heap.data[left].first > heap.data[largest].first
    largest = left
  end

  if right <= length(heap.data) && heap.data[right].first > heap.data[largest].first
    largest = right
  end

  if largest != i
    heap.data[i], heap.data[largest] = heap.data[largest], heap.data[i]
    heapify!(heap, largest)
  end
end

function _k_closest(
  db::AbstractVector{V},
  query::AbstractVector{T},
  k::Int;
  startind::Int=1,
) where {T<:Integer,V<:AbstractVector{T}}
  heap = MaxHeap(k)
  @inbounds for i in eachindex(db)
    d = hamming_distance(db[i], query)
    insert!(heap, d => startind + i - 1)
  end
  return heap.data
end

function k_closest(
  db::AbstractVector{V},
  query::AbstractVector{T},
  k::Int;
  startind::Int=1,
) where {T<:Integer,V<:AbstractVector{T}}
  data = _k_closest(db, query, k; startind=startind)
  return sort!(data, by=x -> x.first)
end

function k_closest_parallel(
  db::AbstractArray{V},
  query::AbstractVector{T},
  k::Int;
  t::Int=nthreads(),
) where {T<:Integer,V<:AbstractVector{T}}
  n = length(db)
  if n < 10_000 || t == 1
    return k_closest(db, query, k)
  end
  task_ranges = [(i:min(i + n ÷ t - 1, n)) for i = 1:n÷t:n]
  tasks = map(task_ranges) do r
    Threads.@spawn _k_closest(view(db, r), query, k; startind=r[1])
  end
  results = fetch.(tasks)
  sort!(vcat(results...), by=x -> x.first)[1:k]
end


function _k_closest(
  db::AbstractMatrix{T},
  query::AbstractVector{T},
  k::Int;
  startind::Int=1,
) where {T<:Integer}
  heap = MaxHeap(k)
  @inbounds for i = 1:size(db, 2)
    d = hamming_distance(view(db, :, i), query)
    insert!(heap, d => startind + i - 1)
  end
  return heap.data
end

function k_closest(
  db::AbstractMatrix{T},
  query::AbstractVector{T},
  k::Int;
  startind::Int=1,
) where {T<:Integer}
  data = _k_closest(db, query, k; startind=startind)
  return sort!(data, by=x -> x.first)
end

function k_closest_parallel(
  db::AbstractMatrix{T},
  query::AbstractVector{T},
  k::Int;
  t::Int=nthreads(),
) where {T<:Integer}
  n = size(db, 2)
  if n < 10_000 || t == 1
    return k_closest(db, query, k)
  end
  task_ranges = [(i:min(i + n ÷ t - 1, n)) for i = 1:n÷t:n]
  tasks = map(task_ranges) do r
    Threads.@spawn _k_closest(view(db, :, r), query, k; startind=r[1])
  end
  results = fetch.(tasks)
  sort!(vcat(results...), by=x -> x.first)[1:k]
end
