#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <math.h>

namespace py = pybind11;


/*
 * \brief Make standard grid from histrogram cubic grid.
 *
 * \param box_size Tuple with box_size
 * \param grid Numpy array to store the grid.
 */
void box_to_std(py::tuple box_size,
                py::array_t<double, py::array::c_style> edges,
                py::array_t<double, py::array::c_style> grid){
    // Get the information from the python objects
    py::buffer_info buf = grid.request(), bufed = edges.request();
    int nx = box_size[0];
    int ny = box_size[1];
    int nz = box_size[2];

    // now make cpp arrays
    double *cgrid = (double *) buf.ptr,
           *cedges = (double *) bufed.ptr;

    int count = 0;
    for(int k=0; k<nz; k++){
        for(int j=0; j<ny; j++){
            for(int i=0; i<nx; i++){
                cgrid[count] = cedges[i];
                cgrid[count+1] = cedges[nx+j];
                cgrid[count+2] = cedges[nx+ny+k];
            }
        }
    }
}

PYBIND11_MODULE(cgrid_tools, m){
    m.doc() = "Tools for grid managing.";
    m.def("box_to_std", &box_to_std,
          "Make histogram cubic grid into standard 3D grid.");
}
