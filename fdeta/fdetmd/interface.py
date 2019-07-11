# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 16:07:43 2015

@author: alaktionov
"""

import numpy as np
import scipy as sp
from scipy import interpolate
import Vis
import copy

class Export:
    """
    Class for communication with another programs.
    Preparation of a PCF to be exported to ADF

    Attributes
    ----------

    guv : dict
        {name of element: 2D numpy array(x,y,z,pcf value,box_grid)}

    """
    BOHR=0.529177249

    def __init__(self):
        self.guv={}
        self.pcf=None
        self.box_size=None
        self.box_size=None
        self.m_id_solute=0
        self.m_id_solvent=1
        self.Rho_Nuclei=None
        self.RhoB=None
        self.RhoNet=None
        self.Electrostatic_potential=None
        self.Electrostatic_potential_ADF=None
        self.Charges=None
        self.delta=None
        self.rhoA=None
        self.Ts_nad=None
        self.Xc_nad=None
        self.Embedding_Potential=None
        self.Electrostatic_potential3DInterpolated=None


    def export_pcf(self,trajectory_analysis_object,molecular_id_solute=0,molecular_id_solvent=1,file_adf_grid=None,box_size=None,box_grid=None):
        """
        This procedure exports PCF to the file which is to be by ADF.
        Steps to be done:
        a) Define Box size
        b) Set a coordinate reference to be the centroid of the solute molecule
        c) Then put the coordinate reference to the center of the Box.
        d) Set the out of the box PCF values to be zero
        e) Import ADF grid and project the PCF onto it. ($ADFHOME/math/intp3d.f90 procedure in adf code)

        Arguments
        ------------
        parameters : list
            List containing the following specifications: a size of the box,

        Info about grid from ADF
        -------------------------
        The BOXSIZE and BOXGRID sub-keys specify dimensions of the simulation box, in Angstrom,
        and the number of points of grid in each direction.
        The box should be twice as large as the molecule and the BOXGRID
        values must be a power of 2. The size/np ratio defines the grid spacing in
        each direction and this should be not larger than 0.5 angstrom.

        """

        self.trajectory_analysis=trajectory_analysis_object
#    Size of a box in which PCF will be stored
        self.box_size=box_size
#    Number of points in each direction
        self.box_grid=box_grid
#    Name of file containing ADF radial grid with weights
        self.adf_grid=file_adf_grid
#    Molecular ID of solute
        self.m_id_solute=molecular_id_solute
#    Molecular ID of solvent
        self.m_id_solvent=molecular_id_solvent
#    Number of atom types of solvent; Only one type of solvent is considered;
        self.trajectory_analysis.select(self.m_id_solvent)
#        Types of atoms of solvent
        self.solvent_atom_types=set(self.trajectory_analysis.Topology[self.m_id_solvent][1])
#        Number of types of atoms of solvent
        self.number_solvent_atom_types=len(self.solvent_atom_types)
#       Generate PCF from MD trajectory
        self.generate_pcf_from_trajectory()
#        Create a structure for guv
        self.create_guv()
#        Convert pcf to guv (On this step we also normilize pcf by dividing it on number of MD frames )
        self.convert_pcf_to_guv()

    def generate_pcf_from_trajectory(self):
        """
            Generate PCF from Trajectory_Analysis object using given box_size and box_grid
            The reference of the coordinate is centered to the centroid of the solute.

        """

#       Range for histogram is taken to equal to the half box size
        self.histogram_range=np.asarray([-self.box_size/2.,self.box_size/2.]).T
#       The box_grid variable contains a number of points in each direction
        self.pcf=self.trajectory_analysis.compute_pair_correlation_functionADF(self.histogram_range,self.box_grid,0)
#        This PCF is generated both for solute and solvent atom types, so we exclude solute types
        for ielement in set(self.pcf.keys()).difference(self.solvent_atom_types):
            del self.pcf[ielement]


    def create_guv(self):
        """
        Create a 3D numpy array in which a PCF for export to ADF will be stored.
        The array is initialzed with zeros and is called guv.
        guv : dict
            guv contains PCF wich is exported to ADF. Key is an atom type:(H,C,N,O....); value: 2D-numpy array([x,y,z,pcf_value])

        """

        if self.box_grid is None:
#        By default step is 0.5A
            self.box_grid=np.round(self.box_size/0.5)
#        Initialize guv with 0
        for k in self.pcf.keys():
            #print "this is (self.box_grid).cumprod()[2]"
            #print (self.box_grid).cumprod()[2]
            self.guv[k]=np.zeros([(self.box_grid).cumprod()[-1],4])
            #print "the whole thing"
            #print np.zeros([(self.box_grid).cumprod()[2],4]).shape


    def convert_pcf_to_guv(self):
        """
            Generate guv via the interpolation of PCF values. Putting it from the boxes on the grid

        """

#        Create arrays to store x,y,z coordinates
        for ielement in self.pcf.keys():
            #-----------New Stuff--------------------
#           We want to keep self.pcf unchanged so one has to perform a deep copy
            H=copy.deepcopy(self.pcf[ielement][0])
            edges=copy.deepcopy(self.pcf[ielement][1])
            #print "H", H
            #print "edges", edges
#            H,edges=self.pcf[ielement]
            #-----------End of New Stuff--------------------
            edges=np.asarray(edges)
            self.delta=sp.diff(self.pcf[ielement][1]) #This expression does not work when Nx!=Ny or Nx!=Nz....
#-----------New Stuff--------------------
#           When Nx!=Ny and so on the dtype of edges is not float64 anymore! dtype=Object
#           delta=np.asarray(map(np.diff,self.pcf[ielement][1]))
#-----------End of the New Stuff---------
            edges=edges[:,:-1]+self.delta/2
#            Norm of PCF (When applied normalisation involve both self.guv and self.pcf)
            H/=self.trajectory_analysis.Total_number_of_frames
#           Divide on an elementary volume (of microcell)
            H/=self.delta[0][0]*self.delta[1][0]*self.delta[2][0]
#            print edges,type(edges)
            Nx,Ny,Nz=H.shape[0],H.shape[1],H.shape[2]
            for k in range(Nz):
                for j in range(Ny):
                    for i in range(Nx):
                        cnt=i+j*Ny+k*Ny*Nz
#                        Loop over x
                        self.guv[ielement][cnt][0]=edges[0][i]
#                        Loop over y
                        self.guv[ielement][cnt][1]=edges[1][j]
#                        Loop over z
                        self.guv[ielement][cnt][2]=edges[2][k]
#                        Loop over PCF value (check consistency!)
                        self.guv[ielement][cnt][3]=H[i][j][k]


    def save_guv(self):
        """
            Save GUV functions to the disk
            First line : size of the box in Angstroms
            Second line : number of points in each direaction
            Third line : centroid of the solute taken from the frame 0 Angstrom
            x,y,z,pcfvalue

        """

        for ielement in self.guv.keys():
            line=' '.join(self.box_size.astype(str))+'\n'+' '.join((self.box_grid).astype(str))+'\n'+' '.join(self.trajectory_analysis.alignement[0][4].astype(str))
            np.savetxt('guv'+ielement,self.guv[ielement],header=line,comments='')


    def compute_rhoB(self,charge_coefficients):
        """
            Computes rhoB
            Density is not normed!

        """
        print "I am in compute_rhoB"
        for ielement in self.pcf.keys():
            #print ielement, self.trajectory_analysis.nametocharge(ielement), self.RhoB is None
            #print "charge_coefficients = ", charge_coefficients[ielement]
            if self.RhoB is None:
                self.RhoB=-charge_coefficients[ielement]*self.trajectory_analysis.nametocharge(ielement)*self.pcf[ielement][0]
            else:
                self.RhoB+=-charge_coefficients[ielement]*self.trajectory_analysis.nametocharge(ielement)*self.pcf[ielement][0]


    def compute_rhoNuclei(self):
        """
            Computes rhoNuclei
            Density is not normed!

        """

        print "I am in compute_rhoNuclei"
        for ielement in self.pcf.keys():
            #print ielement, self.trajectory_analysis.nametocharge(ielement), self.Rho_Nuclei is None
            if self.Rho_Nuclei is None:
                self.Rho_Nuclei=self.trajectory_analysis.nametocharge(ielement)*self.pcf[ielement][0]
            else:
                self.Rho_Nuclei+=self.trajectory_analysis.nametocharge(ielement)*self.pcf[ielement][0]


    def compute_net_rho(self):
        """
            Computes net density: rho_Nuclei+rhoB
            Density is not normed!

        """

        if (self.Rho_Nuclei is not None) and (self.RhoB is not None):
            self.RhoNet=self.Rho_Nuclei+self.RhoB

    def compute_electrostatic_potential(self)            :
        """
            Calculation of an electrostatic potential using net density RhoNet
            For electrostatic potential of solvent we use the same grid as for pcf
            On the contrary, charge distribution is projected on the grid guv.

        """

        BOHR=0.529177249
        if self.RhoNet is None:
            return "The net density has not been generated yet"

        # We use a grid of the first element in a list, because grids for H,C,N,O,.... are the same.
        ielement=self.pcf.keys()[0]
        # The guv grid is taken as the Grid for charges
        # At first, component X is increased, tehn Y then Z; (X,Y,Z values in Angstrom).
        # Putting grid for charges to Charges
        self.Charges=np.insert(self.guv[ielement][:,:3],3,0.0,axis=1)
        # Importing net charges from RhoNet
        Nx,Ny,Nz=self.pcf[ielement][0].shape
        print Nx,Ny,Nz
        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    cnt=i+j*Ny+k*Ny*Nz
                    self.Charges[cnt][3]=self.RhoNet[i][j][k]
        #print "Guv grid length = ", cnt

        # The PCF grid is taken to be a grid for potential; (X,Y,Z values in Angstrom).
        #We assume here that Nx==Ny==Nz==N=Grid_for_Charges+1
        N=np.shape(self.pcf[ielement][1])[1]
        self.Electrostatic_potential=np.zeros((N**3,4))

        for k in range(N):
            for j in range(N):
                for i in range(N):
#                   Setting counter
                    cnt=i+j*N+k*N*N
#                        Loop over x
                    self.Electrostatic_potential[cnt][0]=self.pcf[ielement][1][0][i]
#                        Loop over y
                    self.Electrostatic_potential[cnt][1]=self.pcf[ielement][1][1][j]
#                        Loop over z
                    self.Electrostatic_potential[cnt][2]=self.pcf[ielement][1][2][k]
#                       Projecting RhoNet on the grid for charges

#Loop over charge indicies
        for cnt, iPotentialPoint in enumerate(self.Electrostatic_potential):
#            Get the coordinate for which a value of potential will be computed
            r1=iPotentialPoint[:3]
#            Compute potential at a point r1 (and then convert to a.u.)
            self.Electrostatic_potential[cnt][3]=np.sum(self.Charges[:,3]/np.linalg.norm((self.Charges[:,:3]-r1),axis=1))*BOHR
            #print cnt

#==============================================================================
#             Here we interpolate the values self.Electrostatic_potential
#                 on the rhoB grid to compute embedding potential
        interpolator=interpolate.LinearNDInterpolator(self.Electrostatic_potential[:,:3],self.Electrostatic_potential[:,3])
#        self.Electrostatic_potential3DInterpolated=interpolator(self.guv[ielement][:,:3])
        self.Electrostatic_potential3DInterpolated=interpolator(self.guv[ielement][:,:3]).reshape((Nx,Ny,Nz),order='F')/self.trajectory_analysis.Total_number_of_frames
#==============================================================================
#==============================================================================
#-------------------------------------Data for visualisation----------------------------------------------------------------------
#       Preparation to visualize: pack into a 3D array
        self.ElectrostaticPotential3D=np.zeros((N,N,N))
        for k in range(N):
            for j in range(N):
                for i in range(N):
                    cnt=i+j*N+k*N*N
                    self.ElectrostaticPotential3D[i][j][k]=self.Electrostatic_potential[cnt][3]
#----------------------------We replace the 6 lines above by one line command--------------
#---------------------------pack into a 3D array is Equivalent----------------------------
#        self.ElectrostaticPotential3D=self.Electrostatic_potential[:,3].reshape((N,N,N),order='F')
#==============================================================================
#       x,y,z,ElectrostaticPotential3D
        self.visualisation_data=self.pcf[ielement][1][0],self.pcf[ielement][1][1],self.pcf[ielement][1][2],self.ElectrostaticPotential3D/BOHR
#-------------------------------------End of data for visualisation----------------------------------------------------------------------
#           Cleaning memeory
        del self.Charges


    def save(self,name):
        """
            RhoB,RhoNuclei,RhoNet and guv share the same grid!!!!
            Although the pcf and Electrostatic_potential objects share the same grid,
            their grid differs from those of guv, RhoB,RhoNet,RhoNuclei
            First line : size of the box in Angstroms
            Second line : number of points in each direaction
            Third line : centroid of the solute taken from the frame 0 Angstrom
            x,y,z,value

        """

# We will use the grid for of the first element in pcf structure, because all elements use the same grid
        ielement=self.pcf.keys()[0]
        if name =='RhoB':
            data3D=self.RhoB.copy() # Object to save
            #print "RhoB:"
            #print data3D
            data3D/=self.trajectory_analysis.Total_number_of_frames   # Norm over number of frames
            data3D/=self.delta[0][0]*self.delta[1][0]*self.delta[2][0] # Norm over volume element
            data3D*=Export.BOHR**3                                    # Density in a.u., but grid in angstrom!!!!
            data=np.insert(self.guv[ielement][:,:3],3,0.0,axis=1)     # Variable that holds a grid in angstrom for data3D and zeros as values
        elif name=='RhoNuclei':
            data3D=self.Rho_Nuclei.copy()  # Object to save
            data3D/=self.trajectory_analysis.Total_number_of_frames   # Norm over number of frames
            data3D/=self.delta[0][0]*self.delta[1][0]*self.delta[2][0] # Norm over volume element
            data3D*=Export.BOHR**3                                    # Density in a.u.
            data=np.insert(self.guv[ielement][:,:3],3,0.0,axis=1)     # Final structure to be exported
        elif name=='RhoNet':
            data3D=self.RhoNet.copy()  # Object to save
            data3D/=self.trajectory_analysis.Total_number_of_frames   # Norm over number of frames
            data3D/=self.delta[0][0]*self.delta[1][0]*self.delta[2][0] # Norm over volume element
            data3D*=Export.BOHR**3                                    # Density in au
            data=np.insert(self.guv[ielement][:,:3],3,0.0,axis=1)     # Final structure to be exported
        elif name=='Electrostatic':
            data2D=self.Electrostatic_potential.copy()
            data2D[:,3]/=self.trajectory_analysis.Total_number_of_frames   #Norm
            data=data2D                                                    # Object to save; x,y,z coordinates in angstrom, potential in au (bohr)
        elif name=='ElectrostaticADF':
            data2D=self.Electrostatic_potential_ADF.copy()
            data2D[:,3]/=self.trajectory_analysis.Total_number_of_frames   #Norm
            data=data2D                                                    # Object to save; x,y,z coordinates in Bohr, potential in a.u.

        if (name!='Electrostatic') and (name!='ElectrostaticADF'):
            Nx,Ny,Nz=data3D.shape
    #        Unpacking from 3D array (i,j,k,value) to 2D (x,y,z,value)
            for k in range(Nz):
                for j in range(Ny):
                    for i in range(Nx):
                        cnt=i+j*Ny+k*Ny*Nz
                        data[cnt][3]=data3D[i][j][k]
                        #print data[cnt]
            line=' '.join(self.box_size.astype(str))+'\n'+' '.join((self.box_grid).astype(str))+'\n'+' '.join(self.trajectory_analysis.alignement[0][4].astype(str))
        elif (name=='Electrostatic'):
            line=' '.join(self.box_size.astype(str))+'\n'+' '.join((self.box_grid+1).astype(str))+'\n'+' '.join(self.trajectory_analysis.alignement[0][4].astype(str))
        else:
            line=' '.join((self.box_size/Export.BOHR).astype(str))+'\n'+str(data.shape[0])+'\n'+' '.join((self.trajectory_analysis.alignement[0][4]/Export.BOHR).astype(str))
        np.savetxt(name,data,header=line,comments='')

    def compute_electrostatic_potential_on_adf_grid(self,gridfile='grid')            :
        """
            Calculation of an electrostatic potential using net density RhoNet
            For electrostatic potential of solvent we use the same grid as for pcf
            On the contrary, charge distribution is projected on the grid guv.

        """

        BOHR=0.529177249
        if self.RhoNet is None:
            return "The net density has not been generated yet"

        # We use a grid of the first element in a list, because grids for H,C,N,O,.... are the same.
        ielement=self.pcf.keys()[0]
        # The guv grid is taken as the Grid for charges
        # At first, component X is increased, then Y then Z; (X,Y,Z values in Angstrom).
        # Putting grid for charges to Charges
        self.Charges=np.insert(self.guv[ielement][:,:3],3,0.0,axis=1)
        #print "len charges= ", len(self.Charges)
        # Importing net charges from RhoNet
        Nx,Ny,Nz=self.pcf[ielement][0].shape
        print Nx,Ny,Nz
        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    cnt=i+j*Ny+k*Ny*Nz
                    self.Charges[cnt][3]=self.RhoNet[i][j][k]
#        Converting grid of self.Charges from Angstrom to Bohr
        self.Charges[:,:3]/=BOHR

        # A grid is taken from the file gridname whose structure is :(X,Y,Z, weight) in Bohr.
        self.Electrostatic_potential_ADF=np.genfromtxt(gridfile)
#        We do not use values of weights so they are replaced by zeros
        self.Electrostatic_potential_ADF[:,3]=0.0
#        Centering the adf grid (see rismpot.f90 in ADF) adfgrid+=box_size/2-center_rotation
#        self.Electrostatic_potential_ADF[:,:3]+=(self.box_size/2.0-self.trajectory_analysis.alignement[0][4])/Export.BOHR
#        We consider the coordinate system of ADF grid is that of input coordinates
        self.Electrostatic_potential_ADF[:,:3]-=self.trajectory_analysis.alignement[0][4]/Export.BOHR
        N=self.Electrostatic_potential_ADF.shape[0]
        print "ADF grid length = ", N

#Loop over charge indicies
        for cnt, iPotentialPoint in enumerate(self.Electrostatic_potential_ADF):
#            Get the coordinate for which a value of potential will be computed
            r1=iPotentialPoint[:3]
#            Compute potential at a point r1 (in Bohr)
            self.Electrostatic_potential_ADF[cnt][3]=np.sum(self.Charges[:,3]/np.linalg.norm((self.Charges[:,:3]-r1),axis=1))
            #print cnt,'over',N
#           Cleaning memeory
        del self.Charges

    def load_rhoA(self,filename):
        """
            The method loads rhoA from a file filename in which rhoA is stored
            in the following form: x,y,z,value. Data stored in a.u.

        """

        if self.RhoB is not None:
            self.rhoA=np.genfromtxt(filename).reshape(self.RhoB.shape,order='F')

    def compute_Ts_nad(self,rhA=None,rhB=None):
        """
            Given rhoA and rhoB the method computes the TF nonadditive potential.
            Data stored in a.u. J. Phys. Chem.97, 8050 (1993)
            On the same grid as rhoB

        """

        C=2.871
        if (rhA is not None) and (rhB is not None):
            return (5./3)*C*(sp.special.cbrt(rhA+rhB)**2-sp.special.cbrt(rhA)**2)


        if self.rhoA is None:
            return "rhoA has not been loaded!"
        if self.RhoB is None:
            return "RhoB has not been computed!"

        self.Ts_nad=(5./3)*C*(sp.special.cbrt(self.rhoA+self.RhoB)**2-sp.special.cbrt(self.rhoA)**2)


    def compute_Xc_nad(self):
        """
        Given rhoA and rhoB the method computes a XC nonadditive potential.
        Data stored in a.u. J. Phys. Chem.97, 8050 (1993)
        On the same grid as rhoB

        """

        C=(3./4)*(3/np.pi)**(1./3)
        self.Xc_nad=-(4./3)*C*(sp.special.cbrt(np.fabs(self.rhoA)+np.fabs(self.RhoB))-np.fabs(sp.special.cbrt(self.rhoA)))

    def compute_embedding_potential(self):
        """
        Computation of embedding potential on rhoB grid

        """

        self.Embedding_Potential=self.Electrostatic_potential3DInterpolated+self.Ts_nad+self.Xc_nad
