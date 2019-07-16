#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <math.h>

namespace py = pybind11;


class BoxGrid {
    private:
        py::buffer_info edbuff, boxbuff;
        int nx, ny, nz;
        double *cedges;
    public:
        // Virtual destructor
        ~BoxGrid() {};

        BoxGrid(py::array_t<int, py::array::c_style> grid_size,
                py::array_t<double, py::array::c_style> edges);
        virtual py::array_t<double> get_grid(py::array_t<double, py::array::c_style> grid);
        virtual void save_grid(int grid_length, const char *fname);

};

/*
 * \brief BoxGrid Constructor
 *
 *  \param box_size  The size of cubic box in xyz (Angstrom)
 *  \param edges     Points to be used in each direction
 */
BoxGrid::BoxGrid(py::array_t<int, py::array::c_style> grid_size,
                 py::array_t<double, py::array::c_style> edges){
    edbuff = edges.request();
    boxbuff = grid_size.request();
    cedges = (double *) edbuff.ptr;
    int *sizes = (int *) boxbuff.ptr;
    nx = sizes[0];
    ny = sizes[1];
    nz = sizes[2];
};

py::array_t<double> BoxGrid::get_grid(py::array_t<double, py::array::c_style> grid){
    // Get the information from the python objects
    py::buffer_info buf = grid.request();

    // now make cpp arrays
    double *cgrid = (double *) buf.ptr;

    int count = 0;
    for(int k=0; k<nz; k++){
        for(int j=0; j<ny; j++){
            for(int i=0; i<nx; i++){
                cgrid[count] = cedges[i];
                cgrid[count+1] = cedges[nx+j];
                cgrid[count+2] = cedges[nx+ny+k];
                count += 3;
            }
        }
    }
    return grid;
}

/*
 * \brief Make standard grid from histrogram cubic grid.
 *
 * \param box_size Tuple with box_size
 * \param grid_length Total length of grid.
 */
void BoxGrid::save_grid(int grid_length, const char *fname){
    // now make cpp arrays
    double *cgrid;
    cgrid = new double [grid_length];
    // Create and open a file to print the grid
    FILE * file;
    file = fopen(fname, "w");

    int count = 0;
    for(int k=0; k<nz; k++){
        for(int j=0; j<ny; j++){
            for(int i=0; i<nx; i++){
                cgrid[count] = cedges[i];
                cgrid[count+1] = cedges[nx+j];
                cgrid[count+2] = cedges[nx+ny+k];
                fprintf(file, "%12.10f \t %12.10f \t %12.10f \n",
                        cgrid[count], cgrid[count+1], cgrid[count+2]);
                //printf("%d ", count);
                count += 3;
            }
        }
    }
    // Close grid file
    fclose(file);
}

// In case I want to override/use inheritance in Python.
//class PyBoxGrid : public BoxGrid {
//   public:
//      using BoxGrid::BoxGrid;
//
//       void save_grid(py::array_t<double, py::array::c_style> grid,
//                     std::string fname) override {
//     PYBIND11_OVERLOAD_PURE(void, BoxGrid, save_grid, grid, fname);
//    }
//};


PYBIND11_MODULE(cgrid_tools, m){
    m.doc() = "Tools for grid managing.";
    //py::class_<BoxGrid, PyBoxGrid> boxgrid(m, "BoxGrid");
    py::class_<BoxGrid>(m, "BoxGrid")
          .def(py::init<py::array_t<int>, py::array_t<double>>())
          .def("get_grid", &BoxGrid::get_grid)
          .def("save_grid", &BoxGrid::save_grid);
}
