#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include<iostream>
#include <math.h>

namespace py = pybind11;


class BoxGrid {
    private:
        py::buffer_info edbuff, boxbuff;
        int nx, ny, nz;
        double *cedges;
        const double BOHR = 0.529177249;
        virtual double distance(double* r0, double* r1);
        virtual void make_grid(double *cgrid);
    public:
        // Virtual destructor
        ~BoxGrid() {};

        BoxGrid(py::array_t<int, py::array::c_style> grid_size,
                py::array_t<double, py::array::c_style> edges);
        virtual void electrostatic_potential(int npoints, int nframes, const char *ofname,
                                             py::array_t<double, py::array::c_style> chargedens,
                                             py::array_t<double, py::array::c_style> extgrid);
        virtual py::array_t<double> get_grid(py::array_t<double, py::array::c_style> grid);
        virtual py::array_t<double> normalize(int length, int nframes,
                                              py::array_t<double, py::array::c_style> values);
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

/*
 *  \brief Evaluate the distance between two points
 *
 *  \param r0,r1   Points in space (x, y, z)
 */
double BoxGrid::distance(double *r0, double *r1){
    double result = 0;
    for (int i=0; i<3; i++){
        result += (r0[i] - r1[i]) * (r0[i] - r1[i]);
    }
    return sqrt(result);
}

/*
 *  \brief Construct the array in xyz format from cubic form.
 *
 *  \param cgrid    The final array with grid points.
 */
void BoxGrid::make_grid(double *cgrid){
    // Fill out the grid array
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
};

/*
 * \brief Return xyz grid from histrogram.
 *
 * \param grid Numpy array to store the grid.
 */
py::array_t<double> BoxGrid::get_grid(py::array_t<double, py::array::c_style> grid){
    // Get the information from the python objects
    py::buffer_info buf = grid.request();

    // now make cpp arrays
    double *cgrid = (double *) buf.ptr;
    make_grid(cgrid);

    return grid;
}

/*
 * \brief Save grid from histrogram.
 *
 * \param box_size Tuple with box_size
 * \param grid_length Total length of grid.
 */
void BoxGrid::save_grid(int grid_length, const char *fname){
    // now make cpp arrays
    double *cgrid;
    cgrid = new double [grid_length];
    make_grid(cgrid);
    // Open file to print array
    FILE * file;
    file = fopen(fname, "w");

    int count = 0;
    for(int k=0; k<nz; k++){
        for(int j=0; j<ny; j++){
            for(int i=0; i<nx; i++){
                fprintf(file, "%12.10f \t %12.10f \t %12.10f \n",
                        cgrid[count], cgrid[count+1], cgrid[count+2]);
                count += 3;
            }
        }
    }
    // Close grid file
    fclose(file);
    delete [] cgrid;
}

/*
 * \brief Normalize values from histrogram in xyz (grid in bohr).
 *
 * \param length      Total length of grid.
 * \param nframes     Total number of frames from MD.
 * \param values      Values of Rhob from histogram, in Angstrom.
 */
py::array_t<double> BoxGrid::normalize(int length, int nframes,
                                      py::array_t<double, py::array::c_style> values){
    // Get the information from python objects
    py::buffer_info buf1 = values.request();
    double *cvals = (double *) buf1.ptr;

    // Make xyz grid
    int gridlen = length*3/4;
    double *cgrid;
    cgrid = new double [gridlen];
    make_grid(cgrid);

    // New Numpy array
    ssize_t rlen = (ssize_t) length;
    auto result = py::array_t<double>(rlen);
    py::buffer_info buf2 = result.request();
    double *cresult = (double *) buf2.ptr;
    int count = 0, gcount = 0;
    for(int k=0; k<nz; k++){
        for(int j=0; j<ny; j++){
            for(int i=0; i<nx; i++){
                cresult[count] = cgrid[gcount]/BOHR;
                cresult[count+1] = cgrid[gcount+1]/BOHR;
                cresult[count+2] = cgrid[gcount+2]/BOHR;
                cresult[count+3] = cvals[i*ny*nz+j*nz+k]/nframes;
                gcount += 3;
                count += 4;
            }
        }
    }
    delete [] cgrid;

    return result;
};

/*
 * \brief Evaluate electrostatic potential and save it on file.
 *
 * \param npoints           Total number of grid points.
 * \param nframes           Total number of frames from MD.
 * \param ofname            File name where the potential is saved.
 * \param chargedens        Values of total charge density from histogram, in Angstrom.
 * \param extgrid           External grid where the potential is evaluated, in Bohr.
 */
void BoxGrid::electrostatic_potential(int npoints, int nframes, const char *ofname,
                                      py::array_t<double, py::array::c_style> chargedens,
                                      py::array_t<double, py::array::c_style> extgrid){
    // Get the information from python objects
    py::buffer_info buf1 = chargedens.request(), buf2 = extgrid.request();
    double *cvals = (double *) buf1.ptr,
           *cresult = (double *) buf2.ptr;

    // Open file to print array
    FILE * file;
    file = fopen(ofname, "w");

    ssize_t vcount = 0;
    for(ssize_t j=0; j<buf2.size/4; j++){
        double *r0 = new double [3];
        r0[0] = cresult[j+vcount];
        r0[1] = cresult[j+vcount+1];
        r0[2] = cresult[j+vcount+2];
        ssize_t count = 0;
        for(int i=0; i<npoints; i++){
            double *r1 = new double [3];
            double d;
            r1[0] = cvals[i+count];
            r1[1] = cvals[i+count+1];
            r1[2] = cvals[i+count+2];
            d = distance(r0, r1);
            // avoid very short distances
            if (d > 1e-6){
                cresult[j+vcount+3] += cvals[i+count+3]/d;
            }
            count += 3;
        }
        fprintf(file, "%12.10f \t %12.10f \t %12.10f \t %12.10f \n",
                cresult[j+vcount], cresult[j+vcount+1], cresult[j+vcount+2], cresult[j+vcount+3]);
        vcount += 3;
    }

    // Close grid file
    fclose(file);

};

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
          .def("electrostatic_potential", &BoxGrid::electrostatic_potential)
          .def("get_grid", &BoxGrid::get_grid)
          .def("normalize", &BoxGrid::normalize)
          .def("save_grid", &BoxGrid::save_grid);
}
