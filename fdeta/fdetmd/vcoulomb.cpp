#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <math.h>

namespace py = pybind11;


/*
 * \brief Calculate the Coulomb repulsion potential between two densities.
 *
 * \param grid0 Grid where first density and output is evaluated.
 * \param grid1 Integration grid where second density is evaluated.
 * \param weigths1 Integration weights for grid1.
 * \param density1 Density evaluated at grid1.
 */
py::array_t<double> coulomb_potential(py::array_t<double, py::array::c_style> grid0,
                       py::array_t<double, py::array::c_style> grid1,
                       py::array_t<double, py::array::c_style> weights1,
                       py::array_t<double, py::array::c_style> density1){
    // Get the information from the python arrays
    py::buffer_info buf0 = grid0.request(), buf1 = grid1.request();
    py::buffer_info bufdens1 = density1.request();
    py::buffer_info bufw = weights1.request();
    // Make output array
    auto output = py::array_t<double>(buf0.shape[0]);
    py::buffer_info bufout = output.request();

    // now make cpp arrays
    double *cgrid0 = (double *) buf0.ptr,
           *cgrid1 = (double *) buf1.ptr,
           *cdens1 = (double *) bufdens1.ptr,
           *cweights1 = (double *) bufw.ptr,
           *coutput = (double *) bufout.ptr;

    for(int i=0; i<buf0.shape[0]; i++){
        // Loop over grid1 and integrate
        for(int j=0; j<buf1.shape[0]; j++){
            std::array<double, 3> d;
            // Check something 
            d[0] = pow(cgrid0[i*3] - cgrid1[j*3], 2);
            d[1] = pow(cgrid0[i*3+1] - cgrid1[j*3+1], 2);
            d[2] = pow(cgrid0[i*3+2] - cgrid1[j*3+2], 2);
            double distance = sqrt(d[0] + d[1] + d[2]);
            if (distance > 1e-5){ // avoid very short distances
                coutput[i] += cweights1[j]*cdens1[j]/distance;
            }
        }
    }
    return output;
}

PYBIND11_MODULE(vcoulomb, m){
    m.doc() = "Evaluate the coulomb potential from two densities.";
    m.def("coulomb_potential", &coulomb_potential,
          "The coulomb repulsion between two electronic densities.");
}
