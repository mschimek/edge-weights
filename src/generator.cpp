#include <iostream>
#include <random>

#include <mpi.h>

#include <kagen/context.h>
#include <kagen/edge_range.h>
#include <kagen/io/graph_format.h>
#include <kagen/io/parhip.h>
#include <kagen/kagen.h>

#include <ips4o/ips4o.hpp>

#include "CLI_mpi.hpp"
#include "generate_weights.hpp"

template <typename Write> void write_graph(Write &&write, int rank, int size) {
  bool continue_write = false;
  int round = 0;
  do {
    for (int i = 0; i < size; ++i) {
      if (i == rank) {
        continue_write = write(round);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    ++round;
  } while (continue_write);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  std::string option_string;
  std::string output_file;
  gratr::WeightRange weight_range;
  CLI::App app{"Graph Transformation"};
  app.add_option("--kagen-option-string", option_string,
                 "Options passed verbatim to KaGen.");
  app.add_option("--output-file", output_file,
                 "Path where modified graph is written.");
  app.add_option("--weight-range-begin", weight_range.first,
                 "Begin of weight range.");
  app.add_option("--weight-range-end", weight_range.second,
                 "End of weight range (exclusive).");
  CLI11_PARSE_MPI(app, argc, argv);
  if(!gratr::is_valid(weight_range)) {
    throw std::invalid_argument("invalid weight range");
  }

  int size;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  kagen::KaGen generator(MPI_COMM_WORLD);
  generator.UseCSRRepresentation();
  auto graph = generator.GenerateFromOptionString(option_string);
  kagen::OutputGraphConfig config;
  config.filename = output_file;
  kagen::GraphInfo info(graph, MPI_COMM_WORLD);
  kagen::ParhipWriter writer(config, graph, info, rank, size);
  auto [xadj, adjncy, weights] =
      gratr::generate_weighted_csr_graph(std::move(graph), weight_range, true);

  auto write = [&](int round) {
    return writer.WriteFromCSR<std::uint32_t, std::int32_t, std::int32_t>(
        round, config.filename, xadj, adjncy, nullptr, &weights);
  };
  write_graph(write, 0, 1);
  MPI_Finalize();
}
