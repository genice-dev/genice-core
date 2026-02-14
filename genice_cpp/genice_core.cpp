// genice-core: ice rule を満たす有向グラフを生成（C++ 実装）
// 入出力は配列のみ。API: ice_graph_edges(n_orig, adj, fixed_out, fixed_in, ...) -> vector<pair<int,int>>

#include <algorithm>
#include <optional>
#include <random>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using Adj = std::vector<std::vector<int>>;
using DirectedEdges = std::vector<std::pair<int, int>>;

static int node_to_idx(int v, int n) {
  return v >= 0 ? v : n + (-1 - v);
}
static int idx_to_node(int i, int n) {
  return i < n ? i : -1 - (i - n);
}

static std::vector<std::vector<int>> connected_components(int n_nodes, const Adj& adj) {
  std::vector<bool> seen(n_nodes, false);
  std::vector<std::vector<int>> components;
  for (int v = 0; v < n_nodes; ++v) {
    if (seen[v]) continue;
    std::vector<int> comp;
    std::vector<int> stack = {v};
    seen[v] = true;
    while (!stack.empty()) {
      int cur = stack.back();
      stack.pop_back();
      comp.push_back(cur);
      for (int w : adj[cur]) {
        if (!seen[w]) {
          seen[w] = true;
          stack.push_back(w);
        }
      }
    }
    components.push_back(std::move(comp));
  }
  return components;
}

static void trace_path(int n_nodes, const Adj& adj, std::vector<int>& path,
                       const std::unordered_set<int>& vs) {
  while (true) {
    int last = path[path.size() - 2];
    int head = path[path.size() - 1];
    int next_node = -1;
    for (int w : adj[head]) {
      if (vs.count(w) && w != last) {
        next_node = w;
        break;
      }
    }
    if (next_node < 0) return;
    path.push_back(next_node);
    if (next_node == path[0]) return;
  }
}

static std::vector<int> find_path(int n_nodes, const Adj& adj,
                                  const std::vector<int>& vertex_set) {
  if (vertex_set.empty()) return {};
  std::unordered_set<int> vs(vertex_set.begin(), vertex_set.end());
  int head = vertex_set[0];
  std::vector<int> neighbors;
  for (int w : adj[head]) {
    if (vs.count(w)) neighbors.push_back(w);
  }
  if (neighbors.empty()) return {};
  if (neighbors.size() == 1) {
    std::vector<int> path = {head, neighbors[0]};
    trace_path(n_nodes, adj, path, vs);
    return path;
  }
  std::vector<int> c0 = {head, neighbors[0]};
  trace_path(n_nodes, adj, c0, vs);
  if (c0.back() == head) return c0;
  std::vector<int> c1 = {head, neighbors[1]};
  trace_path(n_nodes, adj, c1, vs);
  std::vector<int> result(c0.rbegin(), c0.rend());
  result.insert(result.end(), c1.begin() + 1, c1.end());
  return result;
}

static void divide(int n_nodes, Adj& adj, int vertex, int offset, std::mt19937& rng) {
  std::vector<int> nei = adj[vertex];
  while (nei.size() < 4u) nei.push_back(-1);
  nei.resize(4);
  std::vector<int> valid;
  for (int x : nei)
    if (x >= 0) valid.push_back(x);
  if (valid.size() < 2u) return;
  std::shuffle(valid.begin(), valid.end(), rng);
  int m0 = valid[0], m1 = valid[1];
  int new_vertex = vertex + offset;
  for (int m : {m0, m1}) {
    auto& am = adj[m];
    am.erase(std::remove(am.begin(), am.end(), vertex), am.end());
    auto& av = adj[vertex];
    av.erase(std::remove(av.begin(), av.end(), m), av.end());
    adj[new_vertex].push_back(m);
    adj[m].push_back(new_vertex);
  }
}

static std::pair<int, Adj> noodlize(int n_orig, const Adj& adj,
                                    const Adj& fixed_out, const Adj& fixed_in) {
  const int n_nodes = 2 * n_orig;
  Adj adj_noodles;
  adj_noodles.reserve(n_nodes);
  for (int v = 0; v < n_orig; ++v) {
    adj_noodles.push_back(adj[v]);
    adj_noodles.back().reserve(4);
  }
  for (int v = 0; v < n_orig; ++v) adj_noodles.emplace_back();
  for (int u = 0; u < n_orig; ++u) {
    int iu = node_to_idx(u, n_orig);
    for (int v : fixed_out[iu]) {
      if (v >= 0 && v < n_orig) {
        auto it = std::find(adj_noodles[u].begin(), adj_noodles[u].end(), v);
        if (it != adj_noodles[u].end()) {
          adj_noodles[u].erase(it);
          auto jt = std::find(adj_noodles[v].begin(), adj_noodles[v].end(), u);
          if (jt != adj_noodles[v].end()) adj_noodles[v].erase(jt);
        }
      }
    }
  }
  std::mt19937 rng(std::random_device{}());
  for (int v = 0; v < n_orig; ++v) {
    int nfixed = (int)fixed_out[node_to_idx(v, n_orig)].size() +
                 (int)fixed_in[node_to_idx(v, n_orig)].size();
    if (nfixed == 0) divide(n_nodes, adj_noodles, v, n_orig, rng);
  }
  return {n_nodes, std::move(adj_noodles)};
}

static std::vector<std::vector<int>> decompose_complex_path(const std::vector<int>& path) {
  std::vector<std::vector<int>> result;
  if (path.empty()) return result;
  std::unordered_map<int, int> order;
  std::vector<int> store;
  order[path[0]] = 0;
  store.push_back(path[0]);
  size_t headp = 1;
  while (headp < path.size()) {
    int node = path[headp];
    if (order.count(node)) {
      int size = (int)order.size() - order[node];
      std::vector<int> cycle(store.end() - size, store.end());
      cycle.push_back(node);
      result.push_back(std::move(cycle));
      for (size_t i = 1; i < result.back().size(); ++i)
        order.erase(result.back()[i]);
      store.resize(store.size() - size);
    }
    order[node] = (int)order.size();
    store.push_back(node);
    ++headp;
  }
  if (store.size() > 1u) result.push_back(std::move(store));
  return result;
}

static std::vector<std::vector<int>> split_into_simple_paths(int n_orig, int n_nodes,
                                                             const Adj& adj) {
  std::vector<std::vector<int>> result;
  auto components = connected_components(n_nodes, adj);
  for (const auto& vertex_set : components) {
    auto path = find_path(n_nodes, adj, vertex_set);
    std::vector<int> flatten;
    for (int v : path) flatten.push_back(v % n_orig);
    auto parts = decompose_complex_path(flatten);
    for (auto& p : parts) result.push_back(std::move(p));
  }
  return result;
}

static void remove_dummy_nodes(int n_orig, Adj& out_adj, Adj& in_adj) {
  int size = n_orig + 4;
  for (int i = n_orig; i < size; ++i) {
    out_adj[i].clear();
    in_adj[i].clear();
  }
  for (int v = 0; v < n_orig; ++v) {
    out_adj[v].erase(
        std::remove_if(out_adj[v].begin(), out_adj[v].end(),
                       [n_orig](int w) { return w < 0 || w >= n_orig; }),
        out_adj[v].end());
    in_adj[v].erase(
        std::remove_if(in_adj[v].begin(), in_adj[v].end(),
                       [n_orig](int u) { return u < 0 || u >= n_orig; }),
        in_adj[v].end());
  }
}

static int choose_free_edge(int n_orig, const Adj& adj, const Adj& out_adj,
                            const Adj& in_adj, int node, std::mt19937& rng) {
  std::vector<int> neis = adj[node];
  while (neis.size() < 4u) neis.push_back(-1 - (int)neis.size());
  neis.resize(4);
  std::shuffle(neis.begin(), neis.end(), rng);
  int i_node = node_to_idx(node, n_orig);
  for (int nei : neis) {
    if (nei == -1) continue;
    int i_nei = nei >= 0 ? node_to_idx(nei, n_orig) : n_orig + (-1 - nei);
    bool has_out = std::find(out_adj[i_node].begin(), out_adj[i_node].end(), nei) !=
                   out_adj[i_node].end();
    bool has_in = std::find(out_adj[i_nei].begin(), out_adj[i_nei].end(), node) !=
                  out_adj[i_nei].end();
    if (!has_out && !has_in) return nei;
  }
  return -999;
}

static std::pair<std::unordered_set<int>, std::unordered_set<int>> get_perimeters(
    int n_orig, const Adj& adj, const Adj& out_adj, const Adj& in_adj) {
  std::unordered_set<int> in_peri, out_peri;
  for (int i = 0; i < n_orig; ++i) {
    int deg_g = (int)adj[i].size();
    int in_d = (int)in_adj[i].size();
    int out_d = (int)out_adj[i].size();
    if (in_d + out_d >= deg_g) continue;
    if (in_d > out_d) out_peri.insert(i);
    else if (in_d < out_d) in_peri.insert(i);
  }
  return {in_peri, out_peri};
}

static std::optional<std::tuple<Adj, Adj, std::vector<std::vector<int>>>>
connect_matching_paths(int n_orig, const Adj& adj, Adj fixed_out, Adj fixed_in,
                      std::mt19937& rng) {
  Adj f_out = fixed_out, f_in = fixed_in;
  auto [in_peri, out_peri] = get_perimeters(n_orig, adj, f_out, f_in);
  std::vector<std::vector<int>> derived_cycles;

  auto add_edge = [&](int u, int v) {
    int iu = node_to_idx(u, n_orig), iv = node_to_idx(v, n_orig);
    f_out[iu].push_back(v);
    f_in[iv].push_back(u);
  };
  auto in_degree = [&](int node) { return (int)f_in[node_to_idx(node, n_orig)].size(); };
  auto out_degree = [&](int node) { return (int)f_out[node_to_idx(node, n_orig)].size(); };

  while (!out_peri.empty()) {
    int node = *out_peri.begin();
    out_peri.erase(node);
    std::vector<int> path = {node};
    int cur = node;
    for (;;) {
      if (cur < 0) break;
      if (in_peri.count(cur)) { in_peri.erase(cur); break; }
      if (out_degree(cur) * 2 > 4) return std::nullopt;
      if ((int)adj[cur].size() == out_degree(cur) + in_degree(cur)) return std::nullopt;
      int next_node = choose_free_edge(n_orig, adj, f_out, f_in, cur, rng);
      if (next_node == -999) return std::nullopt;
      add_edge(cur, next_node);
      if (next_node >= 0) {
        path.push_back(next_node);
        if (in_degree(cur) > out_degree(cur)) out_peri.insert(cur);
      }
      cur = next_node;
      auto it = std::find(path.begin(), path.end() - 1, cur);
      if (it != path.end() - 1) {
        derived_cycles.emplace_back(it, path.end());
        path.erase(it + 1, path.end());
      }
    }
  }

  while (!in_peri.empty()) {
    int node = *in_peri.begin();
    in_peri.erase(node);
    std::vector<int> path = {node};
    int cur = node;
    for (;;) {
      if (cur < 0) break;
      if (out_peri.count(cur)) { out_peri.erase(cur); break; }
      if (out_degree(cur) * 2 > 4) return std::nullopt;
      if ((int)adj[cur].size() == out_degree(cur) + in_degree(cur)) return std::nullopt;
      int next_node = choose_free_edge(n_orig, adj, f_out, f_in, cur, rng);
      if (next_node == -999) return std::nullopt;
      if (next_node >= 0) path.push_back(next_node);
      add_edge(next_node, cur);
      if (next_node >= 0 && in_degree(cur) < out_degree(cur)) in_peri.insert(cur);
      cur = next_node;
      auto it = std::find(path.begin(), path.end() - 1, cur);
      if (it != path.end() - 1) {
        derived_cycles.emplace_back(it, path.end());
        path.erase(it + 1, path.end());
      }
    }
  }

  remove_dummy_nodes(n_orig, f_out, f_in);
  return std::make_tuple(std::move(f_out), std::move(f_in), std::move(derived_cycles));
}

static DirectedEdges arrays_to_directed_edges(int n_orig, const Adj& out_adj,
                                              const Adj& in_adj) {
  DirectedEdges edges;
  size_t cap = 0;
  for (const auto& row : out_adj) cap += row.size();
  edges.reserve(cap);
  for (size_t i = 0; i < out_adj.size(); ++i) {
    int u = idx_to_node((int)i, n_orig);
    if (u < 0) continue;
    for (int v : out_adj[i])
      if (v >= 0 && v < n_orig) edges.emplace_back(u, v);
  }
  return edges;
}

static DirectedEdges path_edges(const std::vector<std::vector<int>>& paths) {
  DirectedEdges out;
  size_t cap = 0;
  for (const auto& p : paths)
    if (p.size() > 1u) cap += p.size() - 1;
  out.reserve(cap);
  for (const auto& p : paths)
    for (size_t i = 0; i + 1 < p.size(); ++i)
      out.emplace_back(p[i], p[i + 1]);
  return out;
}

static DirectedEdges finally_fixed_edges(int n_orig, const Adj& fixed_out,
                                         const Adj& fixed_in,
                                         const std::vector<std::vector<int>>& derived_cycles) {
  auto edges = arrays_to_directed_edges(n_orig, fixed_out, fixed_in);
  std::set<std::pair<int, int>> cycle_set;
  for (const auto& c : derived_cycles)
    for (size_t i = 0; i + 1 < c.size(); ++i) cycle_set.emplace(c[i], c[i + 1]);
  DirectedEdges result;
  for (auto [u, v] : edges)
    if (!cycle_set.count({u, v})) result.emplace_back(u, v);
  return result;
}

DirectedEdges ice_graph_edges(int n_orig, const Adj& adj, Adj fixed_out, Adj fixed_in,
                              int pairing_attempts = 100) {
  Adj f_out = std::move(fixed_out), f_in = std::move(fixed_in);
  std::vector<std::vector<int>> derived_cycles;

  int n_fixed = 0;
  for (const auto& row : f_out) n_fixed += (int)row.size();

  std::mt19937 rng(std::random_device{}());
  if (n_fixed > 0) {
    bool solved = false;
    for (int attempt = 0; attempt < pairing_attempts; ++attempt) {
      auto res = connect_matching_paths(n_orig, adj, f_out, f_in, rng);
      if (res) {
        auto [o, i, dc] = std::move(*res);
        f_out = std::move(o);
        f_in = std::move(i);
        derived_cycles = std::move(dc);
        solved = true;
        break;
      }
    }
    if (!solved) return {};
  }

  DirectedEdges all_edges = finally_fixed_edges(n_orig, f_out, f_in, derived_cycles);
  const auto [n_nodes, adj_noodles] = noodlize(n_orig, adj, f_out, f_in);
  auto paths = split_into_simple_paths(n_orig, n_nodes, adj_noodles);
  for (auto& c : derived_cycles) paths.push_back(std::move(c));

  auto pe = path_edges(paths);
  all_edges.reserve(all_edges.size() + pe.size());
  all_edges.insert(all_edges.end(), pe.begin(), pe.end());
  return all_edges;
}

