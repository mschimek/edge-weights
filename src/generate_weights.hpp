#pragma once

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <random>
#include <span>
#include <vector>

#include <ips4o/parallel.hpp>

#include <kagen/edge_range.h>
#include <kagen/kagen.h>

namespace gratr {
struct WeightedCSR {
  kagen::XadjArray xadj_array;
  std::vector<std::uint32_t> adjncy_array;
  std::vector<std::int32_t> edge_weights;
};

using WeightRange = std::pair<std::uint64_t, std::uint64_t>;

inline bool is_valid(WeightRange weight_range) {
  if (weight_range.first >= weight_range.second) {
    return false;
  }
  if (weight_range.second >= std::numeric_limits<std::int32_t>::max()) {
    return false;
  }
  return true;
}

inline void output_duration(std::string const &desc, double start, double end,
                            bool do_output) {
  if (!do_output) {
    return;
  }
  std::cerr << std::left << std::setw(25) << desc << ":\t" << std::setw(10)
            << (end - start) << " seconds" << std::endl;
}

namespace internal {
using small_edge_t = std::pair<std::uint32_t, std::uint32_t>;

template <typename T> void dump(T &&t) { T{std::move(t)}; }

inline std::vector<small_edge_t> generate_small_edge_list(kagen::Graph graph) {
  std::vector<small_edge_t> edgelist;
  kagen::EdgeRange edge_range(graph);
  edgelist.reserve(edge_range.size());
  for (const auto &edge : edge_range) {
    std::uint32_t u = static_cast<std::int32_t>(edge.first);
    std::uint32_t v = static_cast<std::int32_t>(edge.second);
    edgelist.emplace_back(u, v);
  }
  dump(std::move(graph));
  return edgelist;
}

inline std::vector<std::int32_t>
generate_edge_weighs(std::span<small_edge_t const> edgelist,
                     WeightRange weight_range, bool verbose) {
  std::mt19937 gen;
  std::uniform_int_distribution<std::int32_t> weight_dist(
      weight_range.first, weight_range.second - 1);
  std::int32_t const undefined = -1;
  const double t0 = MPI_Wtime();
  std::vector<std::int32_t> weights(edgelist.size(), undefined);
  for (std::uint64_t index = 0; index < edgelist.size(); ++index) {
    const auto [u, v] = edgelist[index];
    if (u <= v) {
      weights[index] = weight_dist(gen);
    }
  }
  const double t1 = MPI_Wtime();
  output_duration("|-- canoncial weight gen", t0, t1,
                  verbose);
  for (std::uint64_t index = 0; index < edgelist.size(); ++index) {
    const auto [u, v] = edgelist[index];
    if (u > v) {
      auto flipped_edge = std::make_pair(v, u);
      auto it =
          std::lower_bound(edgelist.begin(), edgelist.end(), flipped_edge);
      if (it == edgelist.end() || *it != flipped_edge) {
        throw std::runtime_error("flipped edge not found");
      }
      std::int64_t diff = std::distance(edgelist.begin(), it);
      weights[index] = weights[static_cast<std::uint64_t>(diff)];
    }
  }
  const double t2 = MPI_Wtime();
  output_duration("|-- weight lookup", t1, t2, verbose);
  return weights;
}

inline WeightedCSR generate_csr(std::uint64_t num_vertices,
                                std::span<small_edge_t> edgelist,
                                std::vector<std::int32_t> weights) {
  WeightedCSR csr;
  csr.edge_weights = std::move(weights);
  csr.adjncy_array.reserve(edgelist.size());
  for (const auto [u, v] : edgelist) {
    csr.adjncy_array.emplace_back(v);
  }
  csr.xadj_array.resize(num_vertices + 1, 0);
  for (std::uint64_t u = 0, edge_index = 0; u < num_vertices; ++u) {
    auto index_before = edge_index;
    while (edge_index < edgelist.size() &&
           static_cast<std::uint64_t>(edgelist[edge_index].first) == u) {
      ++edge_index;
    }
    const std::uint64_t degree = edge_index - index_before;
    csr.xadj_array[u + 1] = csr.xadj_array[u] + degree;
  }
  return csr;
}

} // namespace internal
//

inline WeightedCSR generate_weighted_csr_graph(kagen::Graph graph,
                                               WeightRange weight_range,
                                               bool verbose = false) {
  std::uint64_t const num_vertices = graph.NumberOfLocalVertices();
  double t0 = MPI_Wtime();
  auto edgelist = internal::generate_small_edge_list(std::move(graph));
  double t1 = MPI_Wtime();
  output_duration("edgelist generation", t0, t1, verbose);
  ips4o::parallel::sort(edgelist.begin(), edgelist.end());
  double t2 = MPI_Wtime();
  output_duration("edgelist sorting", t1, t2, verbose);
  auto weights =
      internal::generate_edge_weighs(edgelist, weight_range, verbose);
  double t3 = MPI_Wtime();
  output_duration("weight generation", t2, t3, verbose);
  auto csr = internal::generate_csr(num_vertices, edgelist, std::move(weights));
  double t4 = MPI_Wtime();
  output_duration("csr generation", t3, t4, verbose);
  return csr;
}
} // namespace gratr
