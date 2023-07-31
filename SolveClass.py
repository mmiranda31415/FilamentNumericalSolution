# pylint: disable=too-many-instance-attributes
# pylint: disable=trailing-whitespace
# pylint: disable=trailing-newlines
# pylint: disable=line-too-long
# pylint: disable=too-many-arguments
# pylint: disable=invalid-name
# pylint: disable=too-many-ancestors

# import types
import random as rand
import numpy as np
from scipy.signal import find_peaks

### General time evolution classes
# class Mapping, maps a state x -> f(x) = operator @ x + g(x)
# class TimeEvolution, mapping class applied to a modified Crank-Nicholson scheme
# class Solve, repeated use of TimeEvolution methods to yield a list of solutions in a specific time range
### Assumptions: 
# this series of classes assume that 
# - x is an array
# - operator is a matrix
# - g -> g(x) a function of an array

class Mapping:
    """
    Mapping class, maps a state to a new state, such that:
        state -> operator @ state + #(state)
    We assumme that state is a numpy array
    """
    def __init__( self, state, operator ): 
        """
        Initialises the mapping class
        ___
        Paramters:
        state: array like of size n
        operator: array like of size (n,n)

        ___
        Initialises:
        self.state : np.array
        self.operator : np.array
        self.nonlinearity : np.array
        """
        
        self.state = np.array( state )
        self.operator = np.array( operator )
        self.initial_conditions( )
        if not ( operator.shape[0] == operator.shape[1] ) & ( state.shape[0] == operator.shape[0] ):
            # Checks the matrix is square first, then that the matrix and the vector are of same size
            raise ValueError('size mismatch')
        #Here the nonlinearity does not do anything, it is to be implemented in a subclass
        self.nonlinearity = np.zeros( len(self.state) ) 
        self.set_nonlinearity()

        self.state_length = len( self.state )

    def initial_conditions( self ):
        """
        sets_initial_condition, implemented in subclass
        """

    def get_state( self ):
        """Returns state object"""
        return self.state

    
    def mapping_to_new_state( self ):
        """Returns new state object as a function of the old state object"""
        return self.operator @ self.state + self.nonlinearity#( self.state )
        
    def mapping( self ):
        """Maps state object to new state object"""
        self.state = self.mapping_to_new_state( self )

    def set_nonlinearity( self ):
        """
        Changes nonlinearity
        For a non trivial nonlinearity, implement this method again in a subclass
        """
        self.nonlinearity = np.zeros( len(self.state) ) # Does not do anything

    def set_operator( self, operator ):
        """Changes linear operator"""
        self.operator = np.array( operator )

class TimeEvolutionMap( Mapping ):
    """
    Subclass of Mapping class, in which the operator is a time evolution operator, such that
        x_t -> x_(t+ dt) = time_evolution_operator( x_t ) = linear_evolution_operator @ x_t + nonlinear_evolution_function( x_t )
    We assume an evolution equation of the form
        ( x_(t + dt) - x(t) ) / dt = linear_operator @ ( x_(t+ dt) + x_t) / 2 + nonlinear_function( x_t )
    with the 
        linear_evolution_operator = ( 1 - dt / 2 linear_operator)^(-1) ( 1 + dt / 2 linear_operator)
        nonlinear_evolution_function = ( 1 - dt / 2 linear_operator)^(-1) @ nonlinear_function
    """
    def __init__( self, state, operator, dt ):
        super().__init__( state, operator )
        self.dt = dt

        self.forward_operator = np.identity( self.state_length ) + self.dt * self.operator / 2
        self.backward_operator = np.identity( self.state_length ) - self.dt * self.operator / 2

        self.nonlinear_operator = np.linalg.inv( self.backward_operator )
        self.nonlinear_evolution_function = self.nonlinear_operator @ self.nonlinearity#( x )
        self.linear_evolution_operator = self.nonlinear_operator @ self.forward_operator
    
    def advance( self ):
        """Maps state object to a time-evolved state object, then resets an updated nonlinear evolution function"""
        self.state = self.linear_evolution_operator @ self.state + self.nonlinear_evolution_function
        self.reset_nonlinear_evolution_function( )
    
    def get_time_step(self):
        """Returns dt"""
        return self.dt
    

class Solve( TimeEvolutionMap ):
    """
        Subclass of TimeEvolutionMap class.
        This class has the self.solve() method, which applies the self.advance() method repeatedly
    """
    def __init__( self, state, operator, dt, total_time_steps, fraction = 9 / 10):
        super().__init__( state, operator, dt)

        self.total_time_steps = total_time_steps
        self.total_time = self.total_time_steps * self.dt

        self.times = np.arange( self.total_time_steps )

        self.fraction = fraction
        self.append_time = int( fraction * self.total_time_steps )
        self.fraction_times = np.arange( self.total_time_steps - self.append_time )

        self.all_states = np.array( [ self.state for i in self.fraction_times ] )

        self.initial_conditions()

    def initial_conditions( self ):
        """
        sets_initial_condition, implemented in subclass
        """
    
    def dirichlet( self ):
        """
        If a dirichlet boundary has to be implemented this method can be defined in subclass
        """

    def solve( self ):
        """solves the time evolution of the system"""
        
        for time in self.times:
            self.dirichlet()
            self.advance() # updates state and nonlinearity
            self.dirichlet()
            if time >= self.append_time:
                #only saves the last (1 - fraction) states
                #print( str(time - self.append_time) )
                self.all_states[ time - self.append_time ] = self.state
        # self.trim_all_states()
    
    def set_fraction( self, fraction ):
        """Sets new fraction argument"""
        self.fraction = fraction
        self.append_time = int( fraction * self.total_time_steps )
        self.all_states = np.array( [ self.state for i in self.fraction_times ] )

    def get_fracion( self, ):
        """Returns fr"""
        return self.fraction

    def get_all_states(self):
        """
        Returns self.states.
        """
        return self.all_states
    
### Generic Filament Classes
# These classes define some filament specific attributes and methods.
### Assumptions:
# - filament of constant real length filament_length
# - lengthpoints number of grid points along a filament is a constant integer
# - number_of_variables a constant integer
    
class State:
    """
    Defines a state vector

    arguments:
        - number_of_variables : int
        - lengthpoints : int
            Number of grid points along the filament
        - filament_length : float
    """
    def __init__(self, number_of_variables, lengthpoints, filament_length,):
        self.number_of_variables = number_of_variables
        self.lengthpoints = lengthpoints
        self.filament_length = filament_length
        self.dx = self.filament_length / self.lengthpoints
        self.state_length = number_of_variables * lengthpoints

        self.variables = np.arange( self.get_number_of_variables() )

        if self.number_of_variables == 0 :
            raise ValueError("number_of_variables too small")
        else:
            self.state = np.zeros( self.state_length )

    def __call__(self):
        return self.state

    def __str__(self):
        string = (
            f"State of system with" 
            f"{self.number_of_variables}" 
            f"variables of {self.lengthpoints} dimension"
            )
        return string

    def get_number_of_variables(self):
        """Returns number_of_variables"""
        return self.number_of_variables
    
    def get_lengthpoints(self):
        """Returns lengthpoints"""
        return self.lengthpoints

    def get_filament_length(self):
        """Returns filament length"""
        return self.filament_length

    def get_length_step(self):
        """Returns length step"""
        return self.dx

    def get_nth_variable_from_state( self, state, variable ):
        """Returns nth variable of arbitrary state variable"""
        if len( state ) != self.state_length:
            print("size of wrong size")
            return state
        if variable < 0:
            print("invalid  variable < 0.\\Nothing was changed.")
            return state
        if variable > self.number_of_variables:
            print(
                f"invalid  variable > number_of_variables = {self.number_of_variables}."
                "\\Nothing was changed."
                )
            return state
        return state[ variable * self.lengthpoints : ( variable + 1 ) * self.lengthpoints : ]

    def get_nth_variable( self, variable ):
        """Returns nth variable of state attribute"""
        return self.get_nth_variable_from_state( self.state, variable )

    def set_nth_variable( self, variable, new_nth_variable ):
        """Method to change the nth variable"""
        if not len(new_nth_variable) == self.lengthpoints:
            print(
                f"incorrect length of new_variable = {len(new_nth_variable)}" 
                f"not equal to lengthpoints = {self.lengthpoints}."
                "\\Nothing was changed."
                )
        elif variable < 0:
            print("invalid  variable < 0.\\Nothing was changed.")
        elif variable + 1 > self.number_of_variables:
            print(
                "invalid  variable > number_of_variables = "
                f"{self.number_of_variables}."
                "\\Nothing was changed."
                )
        else:
            self.state[ 
                variable * self.lengthpoints : 
                ( variable + 1 ) * self.lengthpoints : 
                ] = new_nth_variable
    
    def random_initial_condition_real_number(self : object, variable : int, var : float = 0.1):
        """
        Sets intial condition for a variable with real numbers, 
        with small curvature and zero curvature at both end
        """
        initial_condition = np.zeros( self.lengthpoints )
        initial_condition[0] = 0
        for i in range( self.lengthpoints - 3 ):
            initial_condition[i+2] = rand.gauss(
                2 * initial_condition[i+1] - initial_condition[i], 
                var * self.dx**2
                )
        initial_condition[-1] = 2 * initial_condition[-2] - initial_condition[-3]
        
        self.set_nth_variable( variable, initial_condition )

    def random_initial_condition_positive_number(self : object, variable : int, characteristic_value : float = 1., var : float = 0.1):
        """
        Sets intial condition for a variable with 
        positive numbers around characteristic_value = 1 by default
        """
        initial_condition = np.zeros( self.lengthpoints )
        for i in range( 1, self.lengthpoints - 1 ):
            initial_condition[i] = rand.gauss(0, var * self.dx**2)
        min_ic = min(initial_condition)
        initial_condition = initial_condition + abs(min_ic) + characteristic_value
        initial_condition[ 0 ] = characteristic_value
        initial_condition[ - 1 ] = characteristic_value
        self.set_nth_variable( variable, initial_condition )
    
    def compute_psi_from_state( self, state, variable ):
        """
        Returns the arbitrary state variable S minus it's basal value S[0]
        """
        vector = self.get_nth_variable_from_state( state, variable )
        return vector - vector[0]

    def compute_psi_from_variable_number( self, variable ):
        """
        Returns the state atribute variable S minus it's basal value S[0]
        """
        return self.compute_psi_from_state( self.state, variable )

    def compute_filament_shape_from_state( self, state, variable ):
        """
        returns the integral of the tangent vector from an arbitrary state
        """
        # Creates the tangent angle vector from a shear variable, such that psi = delta - delta(0)
        psi = self.compute_psi_from_state( state, variable )
        # the tangent is the vector ( cos(psi), sin(psi) )
        tangent_x = np.cos(psi)
        tangent_y = np.sin(psi)
        # we now integrate the tangent to find the filament position
        # we first define the lower triangular matrix "triangle"
        triangle = np.tril( np.ones( ( self.lengthpoints, self.lengthpoints ) ) )
        # the matrix multiplication gives the cumulative sums
        cumulative_sum_x = triangle @ tangent_x
        cumulative_sum_y = triangle @ tangent_y
        # multiplying by the grid-space we get the x and y positions
        x_position = cumulative_sum_x * self.dx
        y_position = cumulative_sum_y * self.dx

        return x_position, y_position

    def compute_filament_shape_from_variable_number( self, variable ):
        """
        returns the integral of the tangent vector from the state attribute
        """
        return self.compute_filament_shape_from_state( self.state, variable )

    def compute_filament_shape( self, ):
        """
        To be defined in subclass
        """

class FiniteDifferences(State):
    """
    Creates finite difference matrices

    arguments:
        - number_of_variables : int
        - lengthpoints : int
            Number of grid points along the filament
        - filament_length : float
    """
    # def __init__(self, number_of_variables, lengthpoints, filament_length): 
    # # Because there is no new attribute created it's not necessary to define a new __init__ method 
    # # instead we can inherit the method from the super class State.
    #     super().__init__( number_of_variables, lengthpoints, filament_length )

    def first_difference(self, edge_order = 1):
        """First finite difference, backwards"""
        matrix = np.zeros( ( self.lengthpoints, self.lengthpoints) )
        # First row is forwards, first order error
        if edge_order == 1:
            matrix[ 0 ][ 0 ] = - 1 
            matrix[ 0 ][ 1 ] = + 1 
        else:
            matrix[ 0 ][ 0 ] = - 3 / 2
            matrix[ 0 ][ 1 ] = + 2
            matrix[ 0 ][ 2 ] = - 1 / 2
        # Second row is central, second order error
        matrix[ 1 ][ 0 ] = - 1 / 2 
        matrix[ 1 ][ 2 ] = + 1 / 2 
        # Third row onwards is a backwards finite difference, second order error
        for i in range( 2, self.lengthpoints ):
            for j in range( self.lengthpoints ):
                if i == j:
                    matrix[ i ][ j ] = 3 / 2
                elif i - 1 == j:
                    matrix[ i ][ j ] = - 2 
                elif i - 2 == j:
                    matrix[ i ][ j ] = 1 / 2
        matrix = matrix / self.dx
        return matrix

    def first_difference_central(self, edge_order = 1):
        """First finite difference, backwards"""
        matrix = np.zeros( ( self.lengthpoints, self.lengthpoints) )
        # First row is forwards, first order error
        if edge_order == 1:
            matrix[ 0 ][ 0 ] = - 1 
            matrix[ 0 ][ 1 ] = + 1 
        else:
            matrix[ 0 ][ 0 ] = - 3 / 2
            matrix[ 0 ][ 1 ] = + 2
            matrix[ 0 ][ 2 ] = - 1 / 2
        for i in range( 1, self.lengthpoints - 1 ):
            matrix[ i ][ i + 1 ] = + 1 / 2

            matrix[ i ][ i - 1 ] = - 1 / 2
        if edge_order == 1:
            matrix[ - 1 ][ - 2 ] = - 1 
            matrix[ - 1 ][ - 1 ] = + 1 
        else:
            matrix[ - 1 ][ - 1 ] = + 3 / 2
            matrix[ - 1 ][ - 2 ] = - 2
            matrix[ - 1 ][ - 3 ] = + 1 / 2
        matrix = matrix / self.dx
        return matrix

    def second_difference(self, edge_order = 1):
        """Second finite difference, central"""
        matrix = np.zeros( ( self.lengthpoints, self.lengthpoints) )
        #First row is forwards
        if edge_order == 1:
            matrix[ 0 ][ 0 ] = + 1
            matrix[ 0 ][ 1 ] = - 2
            matrix[ 0 ][ 2 ] = + 1 
        else:
            matrix[ 0 ][ 0 ] = + 2
            matrix[ 0 ][ 1 ] = - 5
            matrix[ 0 ][ 2 ] = + 4
            matrix[ 0 ][ 3 ] = - 1
        #Second row is central
        for i in range( 1, self.lengthpoints - 1 ):
            for j in range( self.lengthpoints ):
                if abs( i - j ) == 1:
                    matrix[ i ][ j ] = 1 
                elif abs( i - j ) == 0:
                    matrix[ i ][ j ] = - 2 
        #Last row is backwards
        if edge_order == 1:
            matrix[ - 1 ][ - 1 ] = + 1
            matrix[ - 1 ][ - 2 ] = - 2
            matrix[ - 1 ][ - 3 ] = + 1 
        else:
            matrix[ - 1 ][ - 1 ] = + 2
            matrix[ - 1 ][ - 2 ] = - 5
            matrix[ - 1 ][ - 3 ] = + 4 
            matrix[ - 1 ][ - 4 ] = - 1
        matrix = matrix / self.dx**2
        return matrix
    
    def third_difference(self):
        """third finite difference, central"""
        matrix = np.zeros( ( self.lengthpoints, self.lengthpoints) )

        matrix[ 0 ][ 0 ] = - 5 / 2
        matrix[ 0 ][ 1 ] = + 9
        matrix[ 0 ][ 2 ] = - 12
        matrix[ 0 ][ 3 ] = + 7
        matrix[ 0 ][ 4 ] = - 3 / 2

        matrix[ 1 ][ 1 ] = - 5 / 2
        matrix[ 1 ][ 2 ] = + 9 
        matrix[ 1 ][ 3 ] = - 12
        matrix[ 1 ][ 4 ] = + 7
        matrix[ 1 ][ 5 ] = - 3 / 2

        for i in range( 2, self.lengthpoints - 2 ):
            matrix[ i ][ i + 2 ] = + 1 / 2
            matrix[ i ][ i + 1 ] = - 1

            matrix[ i ][ i - 1 ] = + 1
            matrix[ i ][ i - 2 ] = - 1 / 2

        matrix[ - 2 ][ - 2 ] = + 5 / 2
        matrix[ - 2 ][ - 3 ] = - 9 
        matrix[ - 2 ][ - 4 ] = + 12
        matrix[ - 2 ][ - 5 ] = - 7
        matrix[ - 2 ][ - 6 ] = + 3 / 2

        matrix[ - 1 ][ - 1 ] = + 5 / 2
        matrix[ - 1 ][ - 2 ] = - 9
        matrix[ - 1 ][ - 3 ] = + 12 
        matrix[ - 1 ][ - 4 ] = - 7
        matrix[ - 1 ][ - 5 ] = + 3 / 2

        return matrix / self.dx**3
    
    def fourth_difference(self):
        """third finite difference, central"""
        matrix = np.zeros( ( self.lengthpoints, self.lengthpoints) )

        matrix[ 0 ][ 0 ] = + 3
        matrix[ 0 ][ 1 ] = - 14
        matrix[ 0 ][ 2 ] = + 26
        matrix[ 0 ][ 3 ] = - 24
        matrix[ 0 ][ 4 ] = + 11
        matrix[ 0 ][ 5 ] = - 2

        matrix[ 1 ][ 1 ] = + 3
        matrix[ 1 ][ 2 ] = - 14
        matrix[ 1 ][ 3 ] = + 26
        matrix[ 1 ][ 4 ] = - 24
        matrix[ 1 ][ 5 ] = + 11
        matrix[ 1 ][ 6 ] = - 2

        for i in range(2, self.lengthpoints - 2 ):
            matrix[ i ][ i + 2 ] = + 1 
            matrix[ i ][ i + 1 ] = - 4
            matrix[ i ][ i + 0 ] = + 6
            matrix[ i ][ i - 1 ] = - 4
            matrix[ i ][ i - 2 ] = + 1

        matrix[ - 2 ][ - 2 ] = + 3
        matrix[ - 2 ][ - 3 ] = - 14
        matrix[ - 2 ][ - 4 ] = + 26
        matrix[ - 2 ][ - 5 ] = - 24
        matrix[ - 2 ][ - 6 ] = + 11
        matrix[ - 2 ][ - 7 ] = - 2

        matrix[ - 1 ][ - 1 ] = + 3
        matrix[ - 1 ][ - 2 ] = - 14
        matrix[ - 1 ][ - 3 ] = + 26
        matrix[ - 1 ][ - 4 ] = - 24
        matrix[ - 1 ][ - 5 ] = + 11
        matrix[ - 1 ][ - 6 ] = - 2

        return matrix / self.dx**4

class BoundaryConditions( FiniteDifferences ):
    """
    Class defining structure of subsequent sublasses
    """
    def __init__( self, number_of_variables, lengthpoints, filament_length, edge_order = 1 ):
        super().__init__( number_of_variables, lengthpoints, filament_length, )

        self.edge_order = edge_order
        
        self.boundary_conditions = np.zeros( ( self.lengthpoints, self.lengthpoints ) ) # placeholder, instantiates the self.boundary_conditions attribute
        self.assign_boundary_conditions() # assignes self.boundary_conditions its actual value
        
        self.operator = np.zeros( ( self.lengthpoints, self.lengthpoints ) ) # placeholder, instantiates the self.operator attribute
        self.assign_operator() # assigns self.operator its actual value
        
    def assign_boundary_conditions( self ):
        """
        Generates a matrix B, such that operator @ B satistifes the boundary conditions
        Here the matrix B is the trivial zeros matrix
        A different matrix may be defined in subclass
        """
        if self.edge_order == 1:
            pass
        self.boundary_conditions = np.zeros( ( self.lengthpoints, self.lengthpoints ) )
        
    def get_boundary_conditions( self ):
        """
        Returns boundary_conditions attribute
        """
        return self.boundary_conditions

    def assign_operator( self ):
        """
        Generates the matrix operator
        Here the operator is the trivial zeros matrix
        A different matrix may be defied in subclass
        """
        if self.edge_order == 1:
            pass
        self.operator = np.zeros( ( self.lengthpoints, self.lengthpoints ) ) @ self.get_boundary_conditions()

    def get_operator( self ):
        """
        Returns operator attribute
        """
        return self.operator

### Union of classes

class SolveState( BoundaryConditions, Solve ):
    """
    Union of classes State and Solve
    """

    def __init__(
        self, 
        number_of_variables, 
        lengthpoints, 
        filament_length, 
        dt, 
        total_time_steps, 
        fraction = 9 / 10,
        edge_order = 1
        ):

        BoundaryConditions.__init__(
            self,
            number_of_variables,
            lengthpoints,
            filament_length,
            edge_order = edge_order
        )

        Solve.__init__(
            self,
            self.state,         # Defined by BoundaryConditions.__init__()
            self.operator,      # Defined by BoundaryConditions.__init__()
            dt,
            total_time_steps,
            fraction = fraction,
        )

        self.all_variables_history = self.set_all_variables_history_attribute()
        self.periods = np.zeros( self.number_of_variables )

    def set_all_variables_history_attribute( self ):
        """
        Generates an array of arrays for each variables
        """
        array = np.array( 
            [ 
                [ 
                    self.get_nth_variable( variable ) for i in self.fraction_times
                    ] 
                for variable in self.variables 
                ] 
            ) 
        return array

    def get_variable_history_from_solved_states( self, variable, ):
        """
        Returns nth variable history
        """
        history = np.array([ self.get_nth_variable_from_state( state, variable ) for state in self.get_all_states() ])
        return history

    def update_variable_history( self, variable, ): #Not used for anything
        """
        update nth variable history
        """
        self.all_variables_history[ variable ] = self.get_variable_history_from_solved_states( variable )

    def get_all_variables_history( self, ):
        """
        Returns history of all variables
        """
        histories = np.array( [ self.get_variable_history_from_solved_states( variable ) for variable in self.variables ] )
        return histories

    def update_all_variables_history( self, ):
        """
        updates history of all variables
        """
        self.all_variables_history = self.get_all_variables_history()

    def solve_variables( self, ):
        """
        Solves system, returns variables in different arrays
        """
        self.solve()
        self.update_all_variables_history()

    def variable_history( self, variable, ):
        """
        Returns history of a variable
        """
        return self.all_variables_history[ variable ]

    def psi_history( self, variable ):
        """
        Returns history of variable - variable[0]
        """
        psi = np.array( 
            [ self.compute_psi_from_state( state, variable ) for state in self.get_all_states() ]
            )
        return psi
    
    def filament_shape_history( self, variable ):
        """
        Returns filament shape recontructed from variable
        """
        filament_shape = np.array(
            [ self.compute_filament_shape_from_state( state, variable ) for state in self.get_all_states() ]
            ) 
        return filament_shape

    def compute_periods( self ):
        """
        Compute period for each variable
        """
        periods = np.zeros( self.number_of_variables )

        for variable in self.variables:
            history = self.variable_history( variable )
            history = history - history[ - 1 ]
            norm_history = np.array( [ np.sqrt( np.sum( configuration**2 ) ) for configuration in history ] )
            min_norm = np.amin( norm_history )
            max_norm = np.amax( norm_history )
            norm_height = max_norm - min_norm
            peaks = find_peaks( -1 * norm_history, prominence = ( ( 2 / 3 ) * norm_height, None ) )
            period = np.mean( np.ediff1d( peaks[0] ) ) * self.dt
            periods[variable] = period

        return periods
    
    def get_period( self, variable ):
        """
        Return period of variable
        """
        return self.periods[ variable ]
    
    def get_all_periods( self, ):
        """
        Return all periods
        """
        return self.periods

