#ifndef PARAVIEW_WRITER_HPP
#define PARAVIEW_WRITER_HPP

#include <mpi.h>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstring>

/**
 * ParaView writer for 3D uniform Cartesian grids using collective MPI-IO
 * All ranks write to a single .vti file using MPI-IO
 * Supports binary output, ghost cells, cell-centered data, and time series
 */
class ParaViewWriter {
public:
    enum DataLocation {
        NODE_CENTERED,  // Data at grid points
        CELL_CENTERED   // Data at cell centers
    };  
    
    enum OutputPrecision {
        SINGLE,         // 32-bit float (half file size)
        DOUBLE          // 64-bit double (full precision)
    };  
    
    /** 
     * Constructor
     * @param comm MPI communicator
     * @param global_dims Global grid dimensions [nx, ny, nz] (number of points)
     * @param origin Grid origin [x0, y0, z0]
     * @param spacing Grid spacing [dx, dy, dz]
     */
    ParaViewWriter(MPI_Comm comm, 
                   const int global_dims[3],
                   const double origin[3],
                   const double spacing[3])
        : comm_(comm), rank_(-1), nprocs_(-1), output_precision_(SINGLE)
    {   
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &nprocs_);
    
        for (int i = 0; i < 3; ++i) {
            global_dims_[i] = global_dims[i];
            origin_[i]      = origin[i];
            spacing_[i]     = spacing[i];
        }   
    }   
    /** 
     * Set local domain extent for this MPI rank
     * @param local_start Local starting indices [i0, j0, k0] (in global indexing)
     * @param local_dims Local dimensions [ni, nj, nk] (number of points)
     * @param ghost_cells Number of ghost cells on each side [ng_x, ng_y, ng_z]
     */
    void setLocalExtent(const int local_start[3], 
                       const int local_dims[3],
                       const int ghost_cells[3] = nullptr) 
    {
        for (int i = 0; i < 3; ++i) {
            local_start_[i] = local_start[i];
            local_dims_[i] = local_dims[i];
            local_end_[i] = local_start[i] + local_dims[i];
            ghost_cells_[i] = ghost_cells ? ghost_cells[i] : 0;

            // Full dimensions including ghost cells
            local_dims_with_ghost_[i] = local_dims[i] + 2 * ghost_cells_[i];
        }
    }

    /**
     * Set output precision (SINGLE or DOUBLE)
     * SINGLE precision cuts file size in half with minimal visual difference
     */
    void setOutputPrecision(OutputPrecision precision) {
        output_precision_ = precision;
    }
    /**
     * Print diagnostic information about the grid decomposition
     */
    void printDiagnostics() {
        // Gather all rank information to rank 0
        std::vector<int> all_starts, all_dims, all_ends, all_ghosts;

        if (rank_ == 0) {
            all_starts.resize(nprocs_ * 3);
            all_dims.resize(nprocs_ * 3);
            all_ends.resize(nprocs_ * 3);
            all_ghosts.resize(nprocs_ * 3);
        }

        MPI_Gather(local_start_, 3, MPI_INT, all_starts.data(), 3, MPI_INT, 0, comm_);
        MPI_Gather(local_dims_, 3, MPI_INT, all_dims.data(), 3, MPI_INT, 0, comm_);
        MPI_Gather(local_end_, 3, MPI_INT, all_ends.data(), 3, MPI_INT, 0, comm_);
        MPI_Gather(ghost_cells_, 3, MPI_INT, all_ghosts.data(), 3, MPI_INT, 0, comm_);

        if (rank_ == 0) {
            std::cout << "=== ParaView Writer Diagnostics ===" << std::endl;
            std::cout << "Global dims: " << global_dims_[0] << " x "
                     << global_dims_[1] << " x " << global_dims_[2] << std::endl;
            std::cout << "Total points: " << (global_dims_[0] * global_dims_[1] * global_dims_[2]) << std::endl;
            std::cout << "Total cells: " << ((global_dims_[0]-1) * (global_dims_[1]-1) * (global_dims_[2]-1)) << std::endl;
            std::cout << "Number of MPI ranks: " << nprocs_ << std::endl;
            std::cout << "Output precision: " << (output_precision_ == SINGLE ? "SINGLE (Float32)" : "DOUBLE (Float64)") << std::endl;

            for (int r = 0; r < nprocs_; ++r) {
                int idx = r * 3;
                std::cout << "\nRank " << r << ":" << std::endl;
                std::cout << "  Local start: [" << all_starts[idx] << ", "
                         << all_starts[idx+1] << ", " << all_starts[idx+2] << "]" << std::endl;
                std::cout << "  Local dims: [" << all_dims[idx] << ", "
                         << all_dims[idx+1] << ", " << all_dims[idx+2] << "]" << std::endl;
                std::cout << "  Local end: [" << all_ends[idx] << ", "
                         << all_ends[idx+1] << ", " << all_ends[idx+2] << "]" << std::endl;
                std::cout << "  Ghost cells: [" << all_ghosts[idx] << ", "
                         << all_ghosts[idx+1] << ", " << all_ghosts[idx+2] << "]" << std::endl;
                std::cout << "  Local points (no ghost): "
                         << (all_dims[idx] * all_dims[idx+1] * all_dims[idx+2]) << std::endl;
            }
            std::cout << "===================================" << std::endl;
        }
    }

     /**
     * Write single VTI file using collective MPI-IO
     * @param filename Base filename (without extension)
     * @param timestep Optional timestep number (-1 for no timestep)
     * @param time Physical time value (for time series)
     * @param node_scalar_fields Node-centered scalar fields
     * @param node_vector_fields Node-centered vector fields
     */
    void writeVTI(const std::string& filename,
                  int timestep = -1,
                  double time = 0.0,
                  const std::vector<std::pair<std::string, const double*>>& node_scalar_fields = {},
                  const std::vector<std::pair<std::string, const double*>>& node_vector_fields = {})
    {
        std::string base_filename = filename;
        if (timestep >= 0) {
            std::ostringstream oss;
            oss << filename << "_" << std::setfill('0') << std::setw(6) << timestep;
            base_filename = oss.str();
        }

        std::string vti_filename = base_filename + ".vti";

        // Rank 0 writes XML header and footer
        MPI_Offset header_size = 0;
        std::vector<MPI_Offset> field_offsets;

        if (rank_ == 0) {
            header_size = writeXMLHeader(vti_filename,
                                        node_scalar_fields, node_vector_fields,
                                        field_offsets);
        }

        // Broadcast header size and field offsets to all ranks
        MPI_Bcast(&header_size, 1, MPI_OFFSET, 0, comm_);
        size_t n_fields = field_offsets.size();
        MPI_Bcast(&n_fields, 1, MPI_UNSIGNED_LONG, 0, comm_);
        if (rank_ != 0) {
            field_offsets.resize(n_fields);
        }
        MPI_Bcast(field_offsets.data(), n_fields, MPI_OFFSET, 0, comm_);

        // All ranks collectively write their data using MPI-IO
        writeDataCollective(vti_filename, header_size, field_offsets,
                            node_scalar_fields, node_vector_fields);

        // Ensure all ranks have finished writing before rank 0 appends footer
        MPI_Barrier(comm_);

        // Rank 0 appends XML footer
        if (rank_ == 0) {
            writeXMLFooter(vti_filename);

            // Update time series collection if timestep is specified
            if (timestep >= 0) {
                updatePVDCollection(filename, base_filename + ".vti", timestep, time);
            }
        }

        MPI_Barrier(comm_);
    }

private:
    MPI_Comm comm_;
    int rank_, nprocs_;
    int global_dims_[3];
    int local_start_[3], local_dims_[3], local_end_[3];
    int ghost_cells_[3], local_dims_with_ghost_[3];
    double origin_[3], spacing_[3];
    OutputPrecision output_precision_;

    /**
     * Write XML header (rank 0 only)
     * Returns the byte offset where binary data should start
     */
     MPI_Offset writeXMLHeader(const std::string& filename,
                             const std::vector<std::pair<std::string, const double*>>& node_scalar_fields,
                             const std::vector<std::pair<std::string, const double*>>& node_vector_fields,
                             std::vector<MPI_Offset>& field_offsets)
     {
        std::ostringstream header;
        header << std::scientific << std::setprecision(12);

        // VTK header
        header << "<?xml version=\"1.0\"?>\n";
        header << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
        header << "  <ImageData WholeExtent=\""
               << "0 " << global_dims_[0] << " "
               << "0 " << global_dims_[1] << " "
               << "0 " << global_dims_[2] << "\" "
               << "Origin=\"" << origin_[0] << " " << origin_[1] << " " << origin_[2] << "\" "
               << "Spacing=\"" << spacing_[0] << " " << spacing_[1] << " " << spacing_[2] << "\">\n";

        header << "    <Piece Extent=\""
               << "0 " << global_dims_[0] << " "
               << "0 " << global_dims_[1] << " "
               << "0 " << global_dims_[2] << "\">\n";

        // Calculate sizes for each field
        size_t total_points = global_dims_[0] * global_dims_[1] * global_dims_[2];
        size_t total_cells = (global_dims_[0] - 1) * (global_dims_[1] - 1) * (global_dims_[2] - 1);
        size_t bytes_per_value = (output_precision_ == SINGLE) ? sizeof(float) : sizeof(double);

        // Start offset after header (will be updated after writing header)
        MPI_Offset current_offset = 0; // Placeholder, will be set after header

        // Point data (node-centered)
        if (!node_scalar_fields.empty() || !node_vector_fields.empty()) {
            header << "      <PointData>\n";

            for (const auto& field : node_scalar_fields) {
                field_offsets.push_back(current_offset);
                header << "        <DataArray type=\"" << getPrecisionType()
                       << "\" Name=\"" << field.first << "\" format=\"appended\" offset=\""
                       << current_offset << "\"/>\n";
                current_offset += sizeof(uint64_t) + total_points * bytes_per_value;
            }

            for (const auto& field : node_vector_fields) {
                field_offsets.push_back(current_offset);
                header << "        <DataArray type=\"" << getPrecisionType()
                       << "\" Name=\"" << field.first
                       << "\" NumberOfComponents=\"3\" format=\"appended\" offset=\""
                       << current_offset << "\"/>\n";
                current_offset += sizeof(uint64_t) + total_points * 3 * bytes_per_value;
            }

            header << "      </PointData>\n";
        }

        header << "    </Piece>\n";
        header << "  </ImageData>\n";
        header << "  <AppendedData encoding=\"raw\">\n   _";

        // Write header to file
        std::string header_str = header.str();
        std::ofstream file(filename, std::ios::binary);
        file.write(header_str.c_str(), header_str.size());
        file.close();

        return header_str.size();
    }

    /**
     * Write XML footer (rank 0 only)
     */
    void writeXMLFooter(const std::string& filename)
    {
        // Need to append to file, so open in append mode
        std::ofstream file(filename, std::ios::binary | std::ios::app);
        if (!file.is_open()) {
            std::cerr << "Rank 0: Failed to open " << filename << " for footer" << std::endl;
            return;
        }

        file << "\n  </AppendedData>\n";
        file << "</VTKFile>\n";
        file.close();

        // Verify file was written
        std::ifstream check(filename, std::ios::binary | std::ios::ate);
        if (check.is_open()) {
            std::streamsize size = check.tellg();
            check.close();
            std::cout << "Rank 0: Wrote VTI file, total size: " << size << " bytes" << std::endl;
        }
    }
   /**
     * Collectively write data using MPI-IO
     */
    void writeDataCollective(const std::string& filename,
                            MPI_Offset header_size,
                            const std::vector<MPI_Offset>& field_offsets,
                            const std::vector<std::pair<std::string, const double*>>& node_scalar_fields,
                            const std::vector<std::pair<std::string, const double*>>& node_vector_fields)
    {
        MPI_File fh;
        MPI_Status status;

        // Open file for collective writing
        MPI_File_open(comm_, filename.c_str(),
                     MPI_MODE_WRONLY | MPI_MODE_CREATE,
                     MPI_INFO_NULL, &fh);

        size_t field_idx = 0;

        // Write node-centered scalar fields
        for (const auto& field : node_scalar_fields) {
            writeFieldCollective(fh, header_size + field_offsets[field_idx++],
                               field.second, 1, NODE_CENTERED);
        }

        // Write node-centered vector fields
        for (const auto& field : node_vector_fields) {
            writeFieldCollective(fh, header_size + field_offsets[field_idx++],
                               field.second, 3, NODE_CENTERED);
        }

        MPI_File_close(&fh);
    }
 /**
     * Write a single field using collective MPI-IO
     */
    void writeFieldCollective(MPI_File fh,
                             MPI_Offset base_offset,
                             const double* data,
                             int n_components,
                             DataLocation location)
    {
        MPI_Status status;

        // Determine dimensions for this field type
        int global_ni, global_nj, global_nk;
        int local_ni, local_nj, local_nk;

        if (location == NODE_CENTERED) {
            global_ni = global_dims_[0];
            global_nj = global_dims_[1];
            global_nk = global_dims_[2];
            local_ni = local_dims_[0];
            local_nj = local_dims_[1];
            local_nk = local_dims_[2];
        } else { // CELL_CENTERED
            global_ni = global_dims_[0] - 1;
            global_nj = global_dims_[1] - 1;
            global_nk = global_dims_[2] - 1;
            local_ni = local_dims_[0] - 1;
            local_nj = local_dims_[1] - 1;
            local_nk = local_dims_[2] - 1;
        }

        size_t total_size = global_ni * global_nj * global_nk * n_components;

        // Rank 0 writes the size header
        if (rank_ == 0) {
            size_t bytes_per_value = (output_precision_ == SINGLE) ? sizeof(float) : sizeof(double);
            uint64_t byte_size = total_size * bytes_per_value;
            MPI_File_write_at(fh, base_offset, &byte_size, sizeof(uint64_t), MPI_BYTE, &status);
        }

        // Extract interior data (excluding ghost cells)
        size_t local_size = local_ni * local_nj * local_nk * n_components;
        std::vector<double> interior_data(local_size);
        extractInteriorData(data, interior_data.data(), n_components, location);

        size_t bytes_per_value = (output_precision_ == SINGLE) ? sizeof(float) : sizeof(double);

        // Write data slice by slice (in k-j-i order as VTK expects)
        // Each k-slice is a 2D array of size [nj][ni]
        for (int k = 0; k < local_nk; ++k) {
            for (int j = 0; j < local_nj; ++j) {

                // Global indices for this row
                int global_i_start = local_start_[0];
                int global_j = local_start_[1] + j;
                int global_k = local_start_[2] + k;

                // Calculate offset in global array (k varies slowest, i varies fastest)
                size_t global_offset = ((size_t)global_k * global_nj * global_ni +
                                       (size_t)global_j * global_ni +
                                       (size_t)global_i_start) * n_components;

                MPI_Offset file_offset = base_offset + sizeof(uint64_t) +
                                        global_offset * bytes_per_value;

                // Local offset for this row
                size_t local_offset = ((size_t)k * local_nj * local_ni +
                                      (size_t)j * local_ni) * n_components;

                size_t row_size = local_ni * n_components;

                if (output_precision_ == SINGLE) {
                    std::vector<float> float_row(row_size);
                    for (size_t i = 0; i < row_size; ++i) {
                        float_row[i] = static_cast<float>(interior_data[local_offset + i]);
                    }
                    MPI_File_write_at(fh, file_offset, float_row.data(),
                                    row_size, MPI_FLOAT, &status);
                } else {
                    MPI_File_write_at(fh, file_offset, &interior_data[local_offset],
                                    row_size, MPI_DOUBLE, &status);
                }
            }
        }
    }

  /**
     * Extract interior data (excluding ghost cells)
     */
    void extractInteriorData(const double* data,
                            double* output,
                            int n_components,
                            DataLocation location)
    {
        int stride_i = 1;
        int stride_j = local_dims_with_ghost_[0];
        int stride_k = local_dims_with_ghost_[0] * local_dims_with_ghost_[1];

        int nk = (location == CELL_CENTERED) ? local_dims_[2] - 1 : local_dims_[2];
        int nj = (location == CELL_CENTERED) ? local_dims_[1] - 1 : local_dims_[1];
        int ni = (location == CELL_CENTERED) ? local_dims_[0] - 1 : local_dims_[0];

        size_t out_idx = 0;
        for (int k = 0; k < nk; ++k) {
            for (int j = 0; j < nj; ++j) {
                for (int i = 0; i < ni; ++i) {
                     int idx = (i + ghost_cells_[0]) * stride_i +
                               (j + ghost_cells_[1]) * stride_j +
                               (k + ghost_cells_[2]) * stride_k;

                    for (int c = 0; c < n_components; ++c) {
                        output[out_idx++] = data[idx * n_components + c];
                    }
                }
            }
        }
    }

    /**
     * Update PVD collection file for time series
     */
    void updatePVDCollection(const std::string& base_name,
                            const std::string& vti_file,
                            int timestep,
                            double time)
    {
        std::string pvd_filename = base_name + ".pvd";
        std::vector<std::pair<double, std::string>> time_series;

        // Read existing PVD file if it exists
        std::ifstream existing(pvd_filename);
        if (existing.is_open()) {
            std::string line;
            while (std::getline(existing, line)) {
                size_t pos = line.find("timestep=\"");
                if (pos != std::string::npos) {
                    size_t end_pos = line.find("\"", pos + 10);
                    double t = std::stod(line.substr(pos + 10, end_pos - pos - 10));

                    pos = line.find("file=\"");
                    end_pos = line.find("\"", pos + 6);
                    std::string file = line.substr(pos + 6, end_pos - pos - 6);

                    time_series.push_back({t, file});
                }
            }
            existing.close();
        }

        // Add new timestep
        time_series.push_back({time, vti_file});

        // Write updated PVD file
        std::ofstream file(pvd_filename);
        file << "<?xml version=\"1.0\"?>\n";
        file << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        file << "  <Collection>\n";

        for (const auto& entry : time_series) {
            file << "    <DataSet timestep=\"" << entry.first
                 << "\" file=\"" << entry.second << "\"/>\n";
        }

        file << "  </Collection>\n";
        file << "</VTKFile>\n";
        file.close();
    }

    /**
     * Get precision type string
     */
    std::string getPrecisionType() const {
        return (output_precision_ == SINGLE) ? "Float32" : "Float64";
    }
};
#endif // PARAVIEW_WRITER_HPP

// Compile with:
// mpicxx -std=c++11 -O3 -o test_paraview test.cpp
// Run with:
// mpirun -np 4 ./test_paraview
