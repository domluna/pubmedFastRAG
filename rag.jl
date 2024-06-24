using Base.Threads
using StaticArrays
using JSON
using HTTP
using SQLite
using DataFrames
using DBInterface

struct RAGServer
  ids::Vector{Int64}
  data::Matrix{UInt64}
  db::SQLite.DB
end

function RAGServer(dbpath::AbstractString)
  db = SQLite.DB(dbpath)
  ids, data = load_data_from_bin()
  return RAGServer(ids, data, db)
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

function get_article_data(db::SQLite.DB, pmids::Vector{String})
  placeholders = join(fill("?", length(pmids)), ",")
  query = "SELECT pmid, title, authors, abstract, publication_year FROM articles WHERE pmid IN ($placeholders)"

  df = DBInterface.execute(db, query, pmids) |> DataFrame

  article_data = Dict{String,Dict{String,Any}}()
  for row in eachrow(df)
    article_data[string(row.pmid)] = Dict(
      "title" => !ismissing(row.title) ? row.title : "Title not found",
      "authors" =>
        !ismissing(row.authors) ? row.authors : "Authors not found",
      "abstract" =>
        !ismissing(row.abstract) ? row.abstract : "Abstract not found",
      "publication_year" =>
        !ismissing(row.publication_year) ? row.publication_year :
        "Year not found",
    )
  end

  return article_data
end

function start_server(rag::RAGServer; port::Int = 8003)
  router = HTTP.Router()

  # body should {query: str, k: int}
  HTTP.register!(
    router,
    "POST",
    "/find_matches",
    function (req)
      try
        body = JSON.parse(String(req.body))

        # Validate query
        if !haskey(body, "query") ||
           !(body["query"] isa String) ||
           isempty(body["query"])
          return HTTP.Response(
            400,
            "Invalid or missing 'query' parameter. Must be a non-empty string.",
          )
        end
        query_text = body["query"]

        # Validate k
        k = if haskey(body, "k")
          if !(body["k"] isa Integer) || body["k"] <= 0
            return HTTP.Response(
              400,
              "'k' parameter must be a positive integer.",
            )
          end
          body["k"]
        else
          20  # Default value
        end

        # Call embed service
        embed_response = try
          HTTP.post(
            "http://0.0.0.0:8002/embed",
            ["Content-Type" => "application/json"],
            JSON.json(Dict("text" => query_text)),
          )
        catch e
          return HTTP.Response(
            503,
            "Error connecting to embed service: $(sprint(showerror, e))",
          )
        end

        if embed_response.status != 200
          return HTTP.Response(
            502,
            "Error from embed service: $(String(embed_response.body))",
          )
        end

        embed_body = try
          JSON.parse(String(embed_response.body))
        catch e
          return HTTP.Response(
            502,
            "Invalid response from embed service: $(sprint(showerror, e))",
          )
        end

        if !haskey(embed_body, "binary_embedding") ||
           !(embed_body["binary_embedding"] isa Vector)
          return HTTP.Response(
            502,
            "Invalid response format from embed service",
          )
        end

        query = try
          UInt8.(embed_body["binary_embedding"])
        catch e
          return HTTP.Response(
            502,
            "Invalid embedding data from embed service: $(sprint(showerror, e))",
          )
        end

        if length(query) != 64
          return HTTP.Response(
            400,
            "Embedded query must be an array of 64 UInt8 elements",
          )
        end

        query_uint64 = SVector{8,UInt64}(reinterpret(UInt64, query))
        t1 = time()
        results = k_closest_parallel(rag.data, query_uint64, k)
        t2 = time()
        println("Time taken for RAG $(t2 - t1)")

        # Fetch article data from SQLite
        pmids = [string(rag.ids[result.second]) for result in results]
        t1 = time()
        article_data = get_article_data(rag.db, pmids)
        t2 = time()
        println("Time taken for DB query $(t2 - t1)")

        response = [
          Dict(
            "id" => rag.ids[result.second],
            "distance" => result.first,
            "title" =>
              get(article_data, string(rag.ids[result.second]), Dict())["title"],
            "authors" =>
              get(article_data, string(rag.ids[result.second]), Dict())["authors"],
            "abstract" =>
              get(article_data, string(rag.ids[result.second]), Dict())["abstract"],
            "publication_year" =>
              get(article_data, string(rag.ids[result.second]), Dict())["publication_year"],
          ) for result in results
        ]

        return HTTP.Response(200, JSON.json(response))
      catch e
        return HTTP.Response(
          500,
          "Internal server error: $(sprint(showerror, e))",
        )
      end
    end,
  )

  HTTP.serve(router, "0.0.0.0", port)
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
  for i in div(heap.k, 2):-1:1
    heapify!(heap, i)
  end
end

function heapify!(heap::MaxHeap, i::Int)
  left = 2 * i
  right = 2 * i + 1
  largest = i

  if left <= length(heap.data) &&
     heap.data[left].first > heap.data[largest].first
    largest = left
  end

  if right <= length(heap.data) &&
     heap.data[right].first > heap.data[largest].first
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
  startind::Int = 1,
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
  startind::Int = 1,
) where {T<:Integer,V<:AbstractVector{T}}
  data = _k_closest(db, query, k; startind = startind)
  return sort!(data, by = x -> x.first)
end

function k_closest_parallel(
  db::AbstractArray{V},
  query::AbstractVector{T},
  k::Int;
  t::Int = nthreads(),
) where {T<:Integer,V<:AbstractVector{T}}
  n = length(db)
  if n < 10_000 || t == 1
    return k_closest(db, query, k)
  end
  task_ranges = [(i:min(i + n ÷ t - 1, n)) for i in 1:n÷t:n]
  tasks = map(task_ranges) do r
    Threads.@spawn _k_closest(view(db, r), query, k; startind = r[1])
  end
  results = fetch.(tasks)
  sort!(vcat(results...), by = x -> x.first)[1:k]
end


function _k_closest(
  db::AbstractMatrix{T},
  query::AbstractVector{T},
  k::Int;
  startind::Int = 1,
) where {T<:Integer}
  heap = MaxHeap(k)
  @inbounds for i in 1:size(db, 2)
    d = hamming_distance(view(db, :, i), query)
    insert!(heap, d => startind + i - 1)
  end
  return heap.data
end

function k_closest(
  db::AbstractMatrix{T},
  query::AbstractVector{T},
  k::Int;
  startind::Int = 1,
) where {T<:Integer}
  data = _k_closest(db, query, k; startind = startind)
  return sort!(data, by = x -> x.first)
end

function k_closest_parallel(
  db::AbstractMatrix{T},
  query::AbstractVector{T},
  k::Int;
  t::Int = nthreads(),
) where {T<:Integer}
  n = size(db, 2)
  if n < 10_000 || t == 1
    return k_closest(db, query, k)
  end
  task_ranges = [(i:min(i + n ÷ t - 1, n)) for i in 1:n÷t:n]
  tasks = map(task_ranges) do r
    Threads.@spawn _k_closest(view(db, :, r), query, k; startind = r[1])
  end
  results = fetch.(tasks)
  sort!(vcat(results...), by = x -> x.first)[1:k]
end
