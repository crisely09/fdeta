"""Functions to evaluate radial distributions from cubic grids.
"""


import numpy as np
import qcelemental as qce

from fdeta.traj_tools import compute_center_of_mass
from fdeta.fragments import find_fragments, get_interfragment_distances
from fdeta.kabsch import centroid
from fdeta.units import BOHR



def shortest_distance(ref_geo, work_geo):
    """Find the shortest distance between two molecules/fragments.

    """
    distances = get_interfragment_distances(ref_geo, work_geo)
    return min(distances)


def centroid_distance(ref_geo, work_geo):
    """Compute the distance between the centroids of two geometries.

    Parameters
    ----------
    ref_geo : np.ndarray
        Geometry of the reference fragment/molecule.
    work_geo : np.ndarray
        Geometry of the working fragment/molecule.

    Returns
    -------
    distance : float
        Distance between the two centroids.
    """
    ref_centroid = centroid(ref_geo)
    work_centroid = centroid(work_geo)
    return np.linalg.norm(ref_centroid - work_centroid)


def center_of_mass_distance(ref_elements, ref_geo, work_elements, work_geo):
    """Compute the distance between the center of mass of two geometries.

    Parameters
    ----------
    ref_mol : np.ndarray
        Geometry of the reference fragment/molecule.
    work_mol : np.ndarray
        Geometry of the working fragment/molecule.

    Returns
    -------
    distance : float
        Distance between the two centers of mass.
    """
    ref_masses = [atom_to_mass(e) for e in ref_elements]
    ref_center = compute_center_of_mass(ref_masses, ref_geo)
    work_masses = [atom_to_mass(e) for e in work_elements]
    work_center = compute_center_of_mass(work_masses, work_geo)
    return np.linalg.norm(ref_center - work_center)


def compute_pcf(ref_elmts, ref_geos, work_elmts,
                work_geos, dist_type='cofmass'):
    """Compute the pair correlation function between a reference
       molecule/fragment and the rest.

    Parameters
    ----------
    ref_elmts : list or array(str)
        Atomic symbols of the reference molecule/fragment.
    ref_geos : np.ndarray((natoms, 3))
        Geometries of the reference molecule.
    work_elmts : list or array(str)
        Atomic symbols of the rest of molecules/fragmets
    work_geos : np.darray
        Geometries of the rest of the molecules.
    dist_type : str
        Type of distance to use. The options are:
        `cofmass` : use the distance of the center of mass.
        `centroid` : use the distance between the geometric centroid.
        `shortest` : use the shortest distance between the atoms of
        each fragment.
    """
    # Get the all the fragments from the work_geos
    frags = find_fragments(work_elmts, work_geos)[1]
    distances = []
#   if dist_type == ''


def compute_rad(ref_points, grid_values, bins=20, limits=None):
    """From a 3D grid build a radial average distribution.

    Parameters
    ----------
    ref_points : np.ndarray((N, 3))
        Each of N points from where the radial distribution will
        be evaluated
    grid_values : np.ndarray((Nvalues, 4))
        3D grid + value evaluated at each point to be averaged.
    """
    rad_values = []
    points = []
    # Find values limits
    for point in ref_points:
        # First get distances
        ds = get_interfragment_distances(point, grid_values[:, :3])
        ds = np.array(ds)
        if limits is None:
            dmin = min(ds)
            dmax = max(ds)
        else:
            dmin, dmax = limits
        step = (dmax - dmin)/bins
        edges = np.arange(dmin, dmax+step, step)
        # Find corresponing values
        values = []
        eds = []
        for i, edge in enumerate(edges[:-1]):
            end = edges[i+1]
            mask = np.where((edge <= ds) & (ds <= end))[0]
            if mask.any():
                vs = grid_values[mask, 3]
                eds.append(edge+0.5*step)
                values.append(np.mean(vs))
        rad_values.append(values)
        points.append(eds)
    return points, rad_values
