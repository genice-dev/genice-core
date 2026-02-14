# genice-core: ice rule を満たす有向グラフを生成（Julia 実装）
# 入出力は配列のみ。adj は 1-based: adj[i] = ノード i-1 の隣接（0-based ノード番号の Vector）

using Random

# ---------- graph_arrays 相当 ----------
node_to_idx(v, n) = v >= 0 ? v : n + (-1 - v)
idx_to_node(i, n) = i <= n ? i : -1 - (i - n)

# 0-based ノード v の隣接を取得（adj は 1-based）
adj_at(adj, v) = adj[v + 1]

function connected_components(n_nodes, adj)
    seen = falses(n_nodes)
    components = Vector{Int}[]
    for v in 0:(n_nodes - 1)
        if !seen[v + 1]
            comp = Int[]
            stack = [v]
            seen[v + 1] = true
            while !isempty(stack)
                cur = pop!(stack)
                push!(comp, cur)
                for w in adj_at(adj, cur)
                    if !seen[w + 1]
                        seen[w + 1] = true
                        push!(stack, w)
                    end
                end
            end
            push!(components, comp)
        end
    end
    components
end

# ---------- topology: trace_path, find_path ----------
function trace_path!(n_nodes, adj, path, vs)
    while true
        last_, head = path[end-1], path[end]
        next_node = -1
        for w in adj_at(adj, head)
            if w in vs && w != last_
                next_node = w
                break
            end
        end
        next_node < 0 && return
        push!(path, next_node)
        next_node == path[1] && return
    end
end

function find_path(n_nodes, adj, vertex_set)
    isempty(vertex_set) && return Int[]
    vs = Set(vertex_set)
    head = vertex_set[1]
    neighbors = [w for w in adj_at(adj, head) if w in vs]
    isempty(neighbors) && return Int[]
    if length(neighbors) == 1
        path = [head, neighbors[1]]
        trace_path!(n_nodes, adj, path, vs)
        return path
    end
    c0 = [head, neighbors[1]]
    trace_path!(n_nodes, adj, c0, vs)
    c0[end] == head && return c0
    c1 = [head, neighbors[2]]
    trace_path!(n_nodes, adj, c1, vs)
    [reverse(c0); c1[2:end]]
end

# ---------- topology: _divide, noodlize ----------
function divide!(n_nodes, adj, vertex, offset)
    nei = copy(adj_at(adj, vertex))
    while length(nei) < 4
        push!(nei, -1)
    end
    nei = nei[1:4]
    valid = [x for x in nei if x >= 0]
    length(valid) < 2 && return
    shuffle!(valid)
    migrants = [valid[1], valid[2]]
    new_vertex = vertex + offset
    for m in migrants
        a = adj_at(adj, m)
        i = findfirst(==(vertex), a)
        isnothing(i) || deleteat!(a, i)
        a = adj_at(adj, vertex)
        i = findfirst(==(m), a)
        isnothing(i) || deleteat!(a, i)
        push!(adj_at(adj, new_vertex), m)
        push!(adj_at(adj, m), new_vertex)
    end
end

function noodlize(n_orig, adj, fixed_out, fixed_in)
    n_nodes = 2 * n_orig
    adj_noodles = [copy(adj_at(adj, v)) for v in 0:(n_orig-1)]
    append!(adj_noodles, [Int[] for _ in 1:n_orig])
    for u in 0:(n_orig-1)
        iu = node_to_idx(u, n_orig)
        for v in fixed_out[iu + 1]
            if 0 <= v < n_orig
                anu = adj_noodles[u + 1]
                anv = adj_noodles[v + 1]
                i = findfirst(==(v), anu)
                isnothing(i) || deleteat!(anu, i)
                i = findfirst(==(u), anv)
                isnothing(i) || deleteat!(anv, i)
            end
        end
    end
    for v in 0:(n_orig-1)
        i = node_to_idx(v, n_orig)
        nfixed = length(fixed_out[i + 1]) + length(fixed_in[i + 1])
        nfixed == 0 && divide!(n_nodes, adj_noodles, v, n_orig)
    end
    (n_nodes, adj_noodles)
end

# ---------- topology: _decompose_complex_path ----------
function decompose_complex_path(path)
    isempty(path) && return Int[]
    order = Dict{Int,Int}()
    store = Int[]
    order[path[1]] = 0
    push!(store, path[1])
    result = Vector{Int}[]
    headp = 2
    while headp <= length(path)
        node = path[headp]
        if haskey(order, node)
            size = length(order) - order[node]
            cycle = [store[end-size+1:end]; node]
            push!(result, cycle)
            for v in cycle[2:end]
                delete!(order, v)
            end
            store = store[1:end-size]
        end
        order[node] = length(order)
        push!(store, node)
        headp += 1
    end
    length(store) > 1 && push!(result, store)
    result
end

function split_into_simple_paths(n_orig, n_nodes, adj)
    components = connected_components(n_nodes, adj)
    result = Vector{Int}[]
    for vertex_set in components
        path = find_path(n_nodes, adj, vertex_set)
        flatten = [v % n_orig for v in path]
        append!(result, decompose_complex_path(flatten))
    end
    result
end

# ---------- connect_matching_paths 用ヘルパ ----------
function copy_directed(n_orig, out_adj, in_adj)
    size = n_orig + 4
    ([copy(out_adj[i]) for i in 1:size], [copy(in_adj[i]) for i in 1:size])
end

function get_perimeters(n_orig, adj, out_adj, in_adj)
    in_peri = Set{Int}()
    out_peri = Set{Int}()
    for i in 1:n_orig
        node = i - 1
        deg_g = length(adj_at(adj, node))
        in_d = length(in_adj[i])
        out_d = length(out_adj[i])
        in_d + out_d >= deg_g && continue
        in_d > out_d && push!(out_peri, node)
        in_d < out_d && push!(in_peri, node)
    end
    (in_peri, out_peri)
end

function choose_free_edge(n_orig, adj, out_adj, in_adj, node)
    neis = copy(adj_at(adj, node))
    while length(neis) < 4
        push!(neis, -1 - length(neis))
    end
    neis = neis[1:4]
    shuffle!(neis)
    for nei in neis
        nei == -1 && continue
        i_node = node_to_idx(node, n_orig)
        i_nei = nei >= 0 ? node_to_idx(nei, n_orig) : n_orig + (-1 - nei)
        has_out = nei in out_adj[i_node + 1]
        has_in = node in out_adj[i_nei + 1]
        !has_out && !has_in && return nei
    end
    -999
end

function remove_dummy_nodes!(n_orig, out_adj, in_adj)
    for i in (n_orig+1):(n_orig+4)
        empty!(out_adj[i])
        empty!(in_adj[i])
    end
    for v in 0:(n_orig-1)
        out_adj[v+1] = [w for w in out_adj[v+1] if 0 <= w < n_orig]
        in_adj[v+1] = [u for u in in_adj[v+1] if 0 <= u < n_orig]
    end
end

function connect_matching_paths(n_orig, adj, fixed_out, fixed_in)
    (f_out, f_in) = copy_directed(n_orig, fixed_out, fixed_in)
    (in_peri, out_peri) = get_perimeters(n_orig, adj, f_out, f_in)
    derived_cycles = Vector{Int}[]

    add_edge(u, v) = begin
        iu = node_to_idx(u, n_orig)
        iv = node_to_idx(v, n_orig)
        push!(f_out[iu + 1], v)
        push!(f_in[iv + 1], u)
    end
    in_degree(node) = length(f_in[node_to_idx(node, n_orig) + 1])
    out_degree(node) = length(f_out[node_to_idx(node, n_orig) + 1])

    while !isempty(out_peri)
        node = first(out_peri)
        delete!(out_peri, node)
        path = [node]
        cur = node
        while true
            cur < 0 && break
            cur in in_peri && (delete!(in_peri, cur); break)
            out_degree(cur) * 2 > 4 && return nothing
            length(adj_at(adj, cur)) == out_degree(cur) + in_degree(cur) && return nothing
            next_node = choose_free_edge(n_orig, adj, f_out, f_in, cur)
            next_node == -999 && return nothing
            add_edge(cur, next_node)
            if next_node >= 0
                push!(path, next_node)
                in_degree(cur) > out_degree(cur) && push!(out_peri, cur)
            end
            cur = next_node
            idx = findfirst(==(cur), path[1:end-1])
            if !isnothing(idx)
                push!(derived_cycles, path[idx:end])
                path = path[1:idx]
            end
        end
    end

    while !isempty(in_peri)
        node = first(in_peri)
        delete!(in_peri, node)
        path = [node]
        cur = node
        while true
            cur < 0 && break
            cur in out_peri && (delete!(out_peri, cur); break)
            out_degree(cur) * 2 > 4 && return nothing
            length(adj_at(adj, cur)) == out_degree(cur) + in_degree(cur) && return nothing
            next_node = choose_free_edge(n_orig, adj, f_out, f_in, cur)
            next_node == -999 && return nothing
            if next_node >= 0
                push!(path, next_node)
            end
            add_edge(next_node, cur)
            next_node >= 0 && in_degree(cur) < out_degree(cur) && push!(in_peri, cur)
            cur = next_node
            idx = findfirst(==(cur), path[1:end-1])
            if !isnothing(idx)
                push!(derived_cycles, path[idx:end])
                path = path[1:idx]
            end
        end
    end

    remove_dummy_nodes!(n_orig, f_out, f_in)
    (f_out, f_in, derived_cycles)
end

# ---------- directed edges ----------
function arrays_to_directed_edges(n_orig, out_adj, in_adj)
    edges = Tuple{Int,Int}[]
    for i in 1:length(out_adj)
        u = idx_to_node(i - 1, n_orig)
        u < 0 && continue
        for v in out_adj[i]
            (0 <= v < n_orig) && push!(edges, (u, v))
        end
    end
    edges
end

path_edges(paths) = [(p[i], p[i+1]) for p in paths for i in 1:length(p)-1]

function finally_fixed_edges(n_orig, fixed_out, fixed_in, derived_cycles)
    edges = arrays_to_directed_edges(n_orig, fixed_out, fixed_in)
    cycle_set = Set([(c[i], c[i+1]) for c in derived_cycles for i in 1:length(c)-1])
    [(u, v) for (u, v) in edges if (u, v) ∉ cycle_set]
end

# ---------- main ----------
"""
    ice_graph_edges(n_orig, adj, fixed_out, fixed_in; pairing_attempts=100)

Returns `Vector{Tuple{Int,Int}}` of directed edges, or `nothing` if no solution.
adj, fixed_out, fixed_in are 1-based: adj[i] = node i-1's data.
"""
function ice_graph_edges(n_orig, adj, fixed_out, fixed_in; pairing_attempts=100)
    f_out = fixed_out
    f_in = fixed_in
    derived_cycles = Vector{Int}[]

    n_fixed = sum(length, f_out)
    solved = false
    if n_fixed > 0
        for attempt in 1:pairing_attempts
            result = connect_matching_paths(n_orig, adj, f_out, f_in)
            if !isnothing(result)
                (f_out, f_in, derived_cycles) = result
                solved = true
                break
            end
        end
        !solved && return nothing
    end

    fin_fixed = finally_fixed_edges(n_orig, f_out, f_in, derived_cycles)
    (n_nodes, adj_noodles) = noodlize(n_orig, adj, f_out, f_in)
    paths = split_into_simple_paths(n_orig, n_nodes, adj_noodles)
    append!(paths, derived_cycles)

    all_edges = [fin_fixed; path_edges(paths)]
    all_edges
end

# ---------- 使用例 ----------
if abspath(PROGRAM_FILE) == @__FILE__
    Random.seed!(42)
    # adj[i] = ノード i-1 の隣接（0-based 番号）
    adj = [[1, 2], [0, 2], [0, 1], [4], [3]]
    n_orig = 5
    fixed_out = [Int[] for _ in 1:(n_orig+4)]
    fixed_in = [Int[] for _ in 1:(n_orig+4)]

    result = ice_graph_edges(n_orig, adj, fixed_out, fixed_in)
    if !isnothing(result)
        println("edges: ", length(result))
        println("sample: ", result[1:min(5, end)])
    else
        println("no solution")
    end
end
