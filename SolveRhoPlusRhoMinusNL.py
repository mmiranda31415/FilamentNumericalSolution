# pylint: disable=too-many-instance-attributes
# pylint: disable=trailing-whitespace
# pylint: disable=trailing-newlines
# pylint: disable=line-too-long
# pylint: disable=too-many-arguments
# pylint: disable=invalid-name
# pylint: disable=too-many-ancestors

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable as sm
from matplotlib import cm
from matplotlib.colors import ListedColormap

from SolveClass import SolveState

### Now we specify a specific filament model

class BoundaryConditionsRhoPlusRhoMinusNL( SolveState ):
    """
    Creates finite difference matrices wiht boundary conditions specific to the Delta rho_+ rho_- model

    arguments:
        - number_of_variables : int
        - lengthpoints : int
            Number of grid points along the filament
        - filament_length : float
    """
    def __init__( 
        self, 
        number_of_variables, 
        lengthpoints, 
        filament_length, 
        dt, 
        total_time_steps, 
        control_parameter,
        b_squared, 
        xidelta,
        xirho,
        shear_rigidity = 0, 
        basal_rigidity = 0, 
        fraction = 9 / 10, 
        difference = False,
        edge_order = 1,
        ):

        # Model specific parameters
        
        self.control_parameter = control_parameter
        self.b_squared = b_squared
        self.shear_rigidity = shear_rigidity
        self.basal_rigidity = basal_rigidity
        self.xidelta = xidelta
        self.xirho = xirho

        self.difference = difference

        # Inherited methods and attributes are called

        super().__init__( 
            number_of_variables, 
            lengthpoints, 
            filament_length, 
            dt, 
            total_time_steps, 
            fraction = fraction,
            edge_order = edge_order,
            )

    def get_b_squared( self ):
        """
        Returns b_squared
        """
        return self.b_squared

    def get_control_parameter( self ):
        """
        Returns control_parameter
        """
        return self.control_parameter

    def get_shear_rigidity( self ):
        """
        Returns shear_rigidity
        """
        return self.shear_rigidity

    def get_basal_rigidity( self ):
        """
        Returns basal_rigidity
        """
        return self.basal_rigidity

    def compute_boundary_conditions( self, ):
        """
        Generates a matrix B, such that operator @ B satistifes the boundary conditions
        """
        # Delta( dx ) - Delta(0) = self.basal_rigidity * self.dx * Delta( 0 ) - self.b_squared * self.dx * ( RhoPlus( 0 ) - RhoMinus( 0 ) )
        # Delta( L ) - Delta( L - dx ) = - self.b_squared * self.dx * ( RhoPlus( L ) - RhoMinus( L ) )
        
        bc11 = np.identity( self.lengthpoints )
        bc12 = np.zeros( ( self.lengthpoints, self.lengthpoints ) )

        if self.edge_order == 1:
            bc11[ 0 ][ 0 ] = 0
            bc11[ 0 ][ 1 ] = 1 / ( 1 + self.dx * self.basal_rigidity )

            bc11[ - 1 ][ - 1 ] = 0
            bc11[ - 1 ][ - 2 ] = 1

            if self.difference is True:
                bc12[ 0 ][ 0 ] = + self.dx * self.b_squared / ( 1 + self.dx * self.basal_rigidity )
                bc12[ - 1 ][ - 1 ] = - self.dx * self.b_squared
        else:
            bc11[ 0 ][ 0 ] = 0
            bc11[ 0 ][ 1 ] = + ( 4 / 3 ) / ( 1 + 2 / 3 * self.dx * self.basal_rigidity )
            bc11[ 0 ][ 2 ] = - ( 1 / 3 ) / ( 1 + 2 / 3 * self.dx * self.basal_rigidity )

            bc11[ - 1 ][ - 1 ] = 0
            bc11[ - 1 ][ - 2 ] = + 4 / 3
            bc11[ - 1 ][ - 3 ] = - 1 / 3

            if self.difference is True:
                bc12[ 0 ][ 0 ] = + self.dx * self.b_squared / ( 3 / 2 + self.dx * self.basal_rigidity )
                bc12[ - 1 ][ - 1 ] = - self.dx * self.b_squared


        bc13 = - bc12

        bc1 = np.concatenate( ( bc11, bc12, bc13 ), axis = 1)
        
        bc21 = np.zeros( ( self.lengthpoints, self.lengthpoints ) )
        bc22 = np.identity( self.lengthpoints )
        bc23 = np.zeros( ( self.lengthpoints, self.lengthpoints ) )

        bc2 = np.concatenate( ( bc21, bc22, bc23 ), axis = 1)

        bc31 = np.zeros( ( self.lengthpoints, self.lengthpoints ) )
        bc32 = np.zeros( ( self.lengthpoints, self.lengthpoints ) ) 
        bc33 = np.identity( self.lengthpoints )

        bc3 = np.concatenate( ( bc31, bc32, bc33 ), axis = 1)

        matrix = np.concatenate( ( bc1, bc2, bc3 ), axis = 0 )

        return matrix

    def assign_boundary_conditions( self ):
        """
        Assigns boundary condition to attribute self.boundary_conditions
        """
        self.boundary_conditions = self.compute_boundary_conditions( )

    def assign_operator( self) :
        """
        Generates the matrix operator
        """
        matrix11 = - self.shear_rigidity * np.identity( self.lengthpoints ) + self.second_difference( edge_order = self.edge_order )
        matrix12 = + self.control_parameter * np.identity( self.lengthpoints ) + self.b_squared / self.xidelta * self.first_difference_central(  edge_order = self.edge_order )
        matrix13 = - matrix12

        matrix1 = np.concatenate( ( matrix11, matrix12, matrix13 ), axis = 1, )

        matrix21 = - self.b_squared / self.xirho * self.first_difference_central( edge_order = self.edge_order )
        matrix22 = np.zeros( ( self.lengthpoints, self.lengthpoints ) ) 
        matrix23 = np.zeros( ( self.lengthpoints, self.lengthpoints ) ) 

        matrix2 = np.concatenate( ( matrix21, matrix22, matrix23 ), axis = 1, )

        matrix31 = + self.b_squared / self.xirho * self.first_difference_central( edge_order = 2 )
        matrix32 = np.zeros( ( self.lengthpoints, self.lengthpoints ) ) 
        matrix33 = np.zeros( ( self.lengthpoints, self.lengthpoints ) ) 

        matrix3 = np.concatenate( ( matrix31, matrix32, matrix33 ), axis = 1, )

        matrix = np.concatenate( ( matrix1, matrix2, matrix3 ), axis = 0, )

        self.operator = ( self.compute_boundary_conditions( ) @ matrix ) @ self.compute_boundary_conditions( )

        self.matrix11 = self.operator[ 0 : self.lengthpoints, 0 : self.lengthpoints ]
        self.matrix12 = self.operator[ self.lengthpoints : 2 * self.lengthpoints, self.lengthpoints : 2 * self.lengthpoints ]
        self.matrix13 = self.operator[ 2 * self.lengthpoints : :, 2 * self.lengthpoints : : ]


class SolveRhoPlusRhoMinusNL( BoundaryConditionsRhoPlusRhoMinusNL ):
    """
    Solves model with rho_+ and rho_-
    """
    def __init__( 
        self, 
        filament_length = 8,
        control_parameter = 3.8 / 2, 
        b_squared = 0.25, 
        xidelta = 1,
        xirho = 1,
        epsilon = 0.1,
        shear_rigidity = 0, 
        nonlinear_shear_rigidity = 0.1,
        basal_rigidity = 0, 
        lengthpoints = 101,
        dt = 0.06,
        total_time_steps = 10000,
        fraction = 9 / 10,
        difference = True,
        edge_order = 2,
        ):

        self.number_of_variables = 3 # model constant
        self.epsilon = epsilon  #strength of the nonlinearity
        self.nonlinear_shear_rigidity = nonlinear_shear_rigidity
        

        BoundaryConditionsRhoPlusRhoMinusNL.__init__( 
            self,
            self.number_of_variables, 
            lengthpoints, 
            filament_length, 
            dt,
            total_time_steps,
            control_parameter, 
            b_squared, 
            xidelta,
            xirho,
            shear_rigidity = shear_rigidity, 
            basal_rigidity = basal_rigidity,
            fraction = fraction,
            difference = difference,
            edge_order = edge_order,
            )
        
        self.solve_variables() # Solves finite difference equation

        self.delta, self.rhoplus, self.rhominus = self.get_all_variables_history( ) # Solutions of finite difference equation
        self.rho_a = self.rhoplus - self.rhominus
        self.psi = self.psi_history( 0 )
        self.filament_shape = self.filament_shape_history( 0 )

        self.periods = self.compute_periods()

    def get_epsilon( self ):
        """
        Returns epsilon
        """
        return self.epsilon

    def nonlinear_kernel( self, x, ):
        """
        nonlinear function inside nonlinearity
        """
        return - 1 * np.log( 1 + self.epsilon * x ) / self.epsilon

    def set_nonlinearity( self ):
        """
        Actually it's not a function, but a method!
        """
        self.nonlinearity = np.concatenate(
            (
                np.zeros( self.lengthpoints ) - self.nonlinear_shear_rigidity * self.get_nth_variable( 0 )**3,
                self.nonlinear_kernel( self.get_nth_variable( 1 ), ),
                self.nonlinear_kernel( self.get_nth_variable( 2 ), ),
            )
        )
    
    def initial_conditions(self):
        """
        sets_initial_condition, implemented in subclass
        """
        self.random_initial_condition_real_number( 0, )
        self.random_initial_condition_positive_number( 1, characteristic_value = 0.21, var = 0 )
        self.random_initial_condition_positive_number( 2, characteristic_value = 0.21, var = 0 )

    def get_delta( self, t = None ):
        """Returns Delta variable history"""
        if not t is None:
            return self.delta[ t ]
        return self.delta
        

    def get_rhoplus( self, t = None ):
        """Returns rho plus variable history"""
        if not t is None:
            return self.rhoplus[ t ]
        return self.rhoplus

    def get_rhominus( self, t = None ):
        """Returns rho minus variable history"""
        if not t is None:
            return self.rhominus[ t ]
        return self.rhominus

    def get_rho_a(self, t = None):
        """Returns assymetric density"""
        if not t is None:
            return self.rho_a[t]
        return self.rho_a

    def get_psi( self ):
        """
        Returns psi
        """
        return self.psi
    
    def get_filament_shape( self, t = None ):
        """
        Returns filament shape
        """
        if not t is None:
            return self.filament_shape[ t ]
        return self.filament_shape

    def reset_parameters( 
        self,
        filament_length = 8,
        control_parameter = 3.8 / 2, 
        b_squared = 0.25, 
        epsilon = 4,
        shear_rigidity = 0, 
        basal_rigidity = 0, 
        ):
        """
        Reset parameters:
            - filament_length,
            - control_parameter,
            - b_squared,
            - epsilon,
            - shear_rigidity,
            - basal_rigidity,
        by calling self.__init__() again
        """

        self.__init__( 
            filament_length = filament_length,
            control_parameter = control_parameter, 
            b_squared = b_squared, 
            epsilon = epsilon, 
            shear_rigidity = shear_rigidity, 
            basal_rigidity = basal_rigidity,
            )

class Figures:
    """
    Defines some static methods to plot some figures
    """
    @staticmethod
    def plot_delta( ):
        """
        Example method, plots last delta
        """
        filament = SolveRhoPlusRhoMinusNL()
        plt.plot( filament.get_delta()[-1] )

    @staticmethod
    def plot_shape_three_lengths( 
        offsets = ( 10, 5, 0 ), 
        mod = 10, 
        xlim = (0,12), 
        ylim = (-3,12), 
        grey_backround = True,
        **kwargs,
        ):
        """
        Figure of the filament shape for three different filament sizes
        """

        # create figure

        plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        }
        )

        fig, axs = plt.subplots( 1 )

        #create filament instances

        filaments = ( 
            SolveRhoPlusRhoMinusNL( filament_length = 4, **kwargs, ),
            SolveRhoPlusRhoMinusNL( filament_length = 8, **kwargs, ),
            SolveRhoPlusRhoMinusNL( filament_length = 16, **kwargs, ),
            )

        # plot figures

        int_periods = [ int( filament.get_period( 0 ) / filament.dt ) for filament in filaments ]

        colors = [ plt.cm.plasma( np.linspace( 0, 1, p, ), ) for p in int_periods ]

        shapes = [ filament.get_filament_shape() for filament in filaments ]

        if grey_backround is True:
            for t in range( 999 ):
                for shape, offset in zip( shapes, offsets ):
                    axs.plot( shape[ t ][ 0 ], shape[ t ][ 1 ] + offset, 'k', alpha = 0.01)

        for period, shape, offset, color in zip( int_periods, shapes, offsets, colors ):
            for t in np.arange( 0, period, period // mod ):
                axs.plot(  shape[ t ][ 0 ], shape[ t ][ 1 ] + offset, color = color[t] )
        
        axs.axis('square')
        axs.set_xlim( xlim )
        axs.set_ylim( ylim )

        axs.set_xlabel(r'$\bar x$')
        axs.set_ylabel(r'$\bar y$', rotation = 360)

        fig.set_figwidth(3.15)

        maxperiod = max( [ filament.get_period( 0 ) for filament in filaments ] )

        cmap = cm.get_cmap('plasma', 1024)
        cmap = ListedColormap(cmap(np.linspace(0,1,512)))
        norm = plt.Normalize( 0, maxperiod )
        cmap = sm( norm = norm, cmap = cmap )
        cmap.set_array( [ ] )
        colorbar = fig.colorbar( cmap, orientation = 'horizontal', ax = axs,)
        colorbar.set_label(r'$\bar{t}$', rotation = 360)

        return fig, axs