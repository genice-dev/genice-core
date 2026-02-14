# 標準入力から test/data 形式のグラフを読み、有向グラフを標準出力に出す。
# 出力: 1行目 = 有向辺の本数 M、続く M 行が "u v"（tail → head）。解なしは "0" のみ出力して exit 1。

include("genice_core.jl")

function main()
    n_orig = parse(Int, readline())
    m_edges = parse(Int, readline())
    # adj[i] = ノード i-1 の隣接（0-based）。1-based で length n_orig
    adj = [Int[] for _ in 1:n_orig]
    for i in 1:n_orig
        sizehint!(adj[i], 4)  # ice 則グラフは次数 ≤ 4
    end
    for _ in 1:m_edges
        uv = parse.(Int, split(readline()))
        u, v = uv[1] + 1, uv[2] + 1  # 1-based index
        push!(adj[u], uv[2])   # 隣接は 0-based で格納
        push!(adj[v], uv[1])
    end

    n_fixed = parse(Int, readline())
    fixed_out = [Int[] for _ in 1:(n_orig + 4)]
    fixed_in = [Int[] for _ in 1:(n_orig + 4)]
    for _ in 1:n_fixed
        uv = parse.(Int, split(readline()))
        u, v = uv[1] + 1, uv[2] + 1
        push!(fixed_out[u], uv[2])  # 0-based で格納
        push!(fixed_in[v], uv[1])
    end

    # 座標ブロックがあれば読み飛ばす（dipole 未実装のため未使用）
    if !eof(stdin)
        dim = parse(Int, readline())
        for _ in 1:n_orig
            readline()
        end
    end

    result = ice_graph_edges(n_orig, adj, fixed_out, fixed_in)
    if isnothing(result)
        println("0")
        exit(1)
    end
    # パイプ時 (e.g. | wc -l) でブロックバッファによるハングを防ぐため定期的に flush
    println(length(result))
    flush(stdout)
    for (idx, (u, v)) in enumerate(result)
        println(u, " ", v)
        if idx % 10000 == 0
            flush(stdout)
        end
    end
    flush(stdout)
    exit(0)
end

main()
