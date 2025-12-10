#include "paraview_writer.hpp"
#include <cmath>

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    // Use grid that divides evenly: 32^3 with 4 ranks = 8 slices per rank
    int global_dims[3] = {32, 32, 32};
    double origin[3] = {0.0, 0.0, 0.0};
    double spacing[3] = {1.0, 1.0, 1.0};
    
    // Create writer
    ParaViewWriter writer(MPI_COMM_WORLD, global_dims, origin, spacing);
    writer.setOutputPrecision(ParaViewWriter::SINGLE);
    
    // Simple 1D decomposition in Z (ensure even division)
    int nz_per_rank = global_dims[2] / nprocs;
    
    int local_start[3] = {0, 0, rank * nz_per_rank};
    int local_dims[3] = {global_dims[0], global_dims[1], nz_per_rank};
    int ghost_cells[3] = {1, 1, 1}; // Test with ghost cells
    
    writer.setLocalExtent(local_start, local_dims, ghost_cells);
    
    // All ranks participate in diagnostics
    writer.printDiagnostics();
    
    // Allocate local data (WITH ghost cells)
    int nx_ghost = local_dims[0] + 2 * ghost_cells[0];
    int ny_ghost = local_dims[1] + 2 * ghost_cells[1];
    int nz_ghost = local_dims[2] + 2 * ghost_cells[2];
    
    size_t n_points_ghost = nx_ghost * ny_ghost * nz_ghost;
    size_t n_cells_ghost = (nx_ghost - 1) * (ny_ghost - 1) * (nz_ghost - 1); 
    
    // Node-centered data
    std::vector<double> pressure(n_points_ghost);
    std::vector<double> velocity(3 * n_points_ghost);
    
    // Fill with pattern that includes ghost cells
    // For pressure: use global k-index (including offset for ghosts)
    for (int k = 0; k < nz_ghost; ++k) {
        for (int j = 0; j < ny_ghost; ++j) {
            for (int i = 0; i < nx_ghost; ++i) {
                int idx = k * nx_ghost * ny_ghost + j * nx_ghost + i;
    
                // Global k accounting for ghost offset
                int global_k = local_start[2] + (k - ghost_cells[2]);
                pressure[idx] = (double)global_k;

                // Simple velocity pattern
                velocity[3*idx] = 1.0;
                velocity[3*idx+1] = 2.0;
                velocity[3*idx+2] = 3.0 + rank;
            }
        }
    }

    // Write output with both node and cell data
    std::vector<std::pair<std::string, const double*>> node_scalars = {
        {"pressure", pressure.data()}
    };

    std::vector<std::pair<std::string, const double*>> node_vectors = {
        {"velocity", velocity.data()}
    };

    writer.writeVTI("test_ghost", 0, 0.0, node_scalars, node_vectors);

    if (rank == 0) {
        std::cout << "\nTest output written to test_ghost_000000.vti" << std::endl;
        std::cout << "This test includes:" << std::endl;
        std::cout << "  - Multiple MPI ranks: " << nprocs << std::endl;
        std::cout << "  - Ghost cells: 1 layer on each side" << std::endl;
        std::cout << "  - Node-centered: pressure (scalar), velocity (vector)" << std::endl;
        std::cout << "\nTry opening in ParaView and check:" << std::endl;
        std::cout << "  1. Pressure should vary smoothly from 0 to 31 in Z direction" << std::endl;
        std::cout << "  2. No discontinuities at rank boundaries" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
