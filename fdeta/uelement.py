#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  Created on Sept 2020
#  Trajectory Class
#  @author: C.G.E.
"""
Base Class for trajectory analysis.

"""

from fdeta.traj_tools import atom_to_mass
from fdeta.traj_tools import clean_atom_name, atom_to_charge


def get_unique_elements(elements, charges):
    """ Sort charges by atom type.
    It recognizes the different atom types by charges
    and makes the list of list of charges per frame.
    
    Parameters
    ----------
    elements : list
        List of elements per frame.
    charges : dict or list
        Information about charges in trajectory, either
        a dictionary or a list of charges per frame.
    
    """
    luniques = []
    uniques = []
    repeated = {}  # use to count repeated atoms with diff charges
    nframes = len(elements)
    # Case where charges are given in a list per frame
    if isinstance(charges, list):
        if len(elements) != len(charges):
            raise ValueError('Number of frames of `charges` and `elements` do not match')
        for iframe in range(nframes):
            natoms = len(elements[iframe])
            if len(charges[iframe]) != natoms:
                raise ValueError('Number of atoms of `charges` and `elements` do not match')
            for iatom in range(natoms):
                element = elements[iframe][iatom]
                charge = charges[iframe][iatom]
                # check that the element is used already
                if element not in luniques:
                    ename = element
                    uelem = UElement(element, charge)
                else:
                    index = luniques.index(element)
                    # Check if charge is different
                    if charge != uniques[index].charge:
                        if element in repeated:
                            nrep = range(1, repeated[element]+1)
                            rindices = [luniques.index(element+str(i)) for i in nrep]
                            rcharges = [uniques[ind].charge for ind in rindices]
                            if charge not in rcharges:
                                repeated[element] += 1
                                ename = element+str(repeated[element])
                                uelem = UElement(ename, charge)
                            else:
                                cindex = rcharges.index(charge)
                                findex = rindices[cindex]
                                ename = element+str(nrep[cindex])
                                uelem = uniques[findex]
                        else:
                            repeated[element] = 1
                            ename = element+str(repeated[element])
                            uelem = UElement(ename, charge)
                    else:
                        ename = element
                        uelem = uniques[index]
                if uelem.count_frames is None:  # New element
                    uelem.count_frames = [0]*nframes
                    uelem.alloc_traj = []
                    for j in range(nframes):
                        uelem.alloc_traj.append([])
                uelem.count_frames[iframe] += 1
                uelem.total_count += 1
                uelem.alloc_traj[iframe].append(iatom)
                # Finally add it to the list
                if ename not in luniques:
                    uniques.append(uelem)
                    luniques.append(ename)
                del uelem
    elif isinstance(charges, dict):
        for iframe in range(nframes):
            natoms = len(elements[iframe])
            for iatom in range(natoms):
                element = elements[iframe][iatom]
                charge = charges[element]
                # check that the element is used in
                if element not in luniques:
                    uelem = UElement(element, charge)
                else:
                    index = luniques.index(element)
                    # Check if charge is different
                    if charge != uniques[index].charge:
                        if element in repeated:
                            repeated[element] +=1
                        else:
                            repeated[element] = 1
                        name = element+str(repeated[element])
                        uelem = UElement(name, charge)
                    else:
                        uelem = uniques[index]
                if uelem.count_frames is None:  # New element
                    uelem.count_frames = [0]*nframes
                    uelem.alloc_traj = []
                    for j in range(nframes):
                        uelem.alloc_traj.append([])
                uelem.count_frames[iframe] += 1
                uelem.total_count += 1
                uelem.alloc_traj[iframe].append(iatom)
                # Finally add it to the list
                if element not in luniques:
                    uniques.append(uelem)
                    luniques.append(element)
                del uelem
    else:
        raise TypeError('`charges` must be given as list or dictionary')
    print('luniques', luniques)
    return uniques


class UElement:
    """Unique elements.

    Attributes
    ----------
    name : str
    symbol : str
    charge : float
    zcharge : float
    mass : float
    count_frames : list
    alloc_traj : list(list)
    """
    def __init__(self, name, charge, count_frames=None, alloc_traj=None):
        """ Object with all the information about each element.
        """
        if not isinstance(name, str):
            raise TypeError('`name` must be a string.')
        if not isinstance(charge, float):
            raise TypeError('`charge` must be a float.')
        self.name = name
        self.symbol = clean_atom_name(name)
        self.zcharge = atom_to_charge(self.symbol) 
        self.charge = charge
        self.mass = atom_to_mass(self.symbol)
        self.count_frames = count_frames
        self.alloc_traj = alloc_traj
        self.frame_count = 0
        self.total_count = 0
