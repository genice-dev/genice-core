// 標準入力から test/data 形式のグラフを読み、有向グラフを標準出力に出す。
// 出力: 1行目 = 有向辺の本数 M、続く M 行が "u v"。解なしは "0" のみ出力して exit 1。

#include <iostream>
#include <sstream>
#include <string>
#include "genice_core.cpp"

int main() {
  int n_orig, m_edges;
  std::cin >> n_orig >> m_edges;
  Adj adj(n_orig);
  for (std::vector<int>& row : adj)
    row.reserve(4);  // ice 則グラフは次数 ≤ 4
  for (int i = 0; i < m_edges; ++i) {
    int u, v;
    std::cin >> u >> v;
    adj[u].push_back(v);
    adj[v].push_back(u);
  }
  int n_fixed;
  std::cin >> n_fixed;
  Adj fixed_out(n_orig + 4), fixed_in(n_orig + 4);
  for (std::vector<int>& row : fixed_out) row.reserve(2);
  for (std::vector<int>& row : fixed_in) row.reserve(2);
  for (int i = 0; i < n_fixed; ++i) {
    int u, v;
    std::cin >> u >> v;
    fixed_out[u].push_back(v);
    fixed_in[v].push_back(u);
  }
  // 座標ブロックがあれば読み飛ばす（n_fixed 行の後の改行を消費してから次行を読む）
  std::string line;
  std::getline(std::cin, line);
  if (std::getline(std::cin, line) && !line.empty()) {
    int dim = std::stoi(line);
    for (int i = 0; i < n_orig; ++i)
      std::getline(std::cin, line);
  }

  DirectedEdges edges = ice_graph_edges(n_orig, adj, fixed_out, fixed_in);
  if (edges.empty()) {
    std::cout << "0\n";
    return 1;
  }
  std::cout << edges.size() << "\n";
  for (const auto& [u, v] : edges)
    std::cout << u << " " << v << "\n";
  return 0;
}
