from random import randint, random
from functools import lru_cache
import numpy as np
from math import log, pi, exp
from scipy.constants import physical_constants
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

# Constants definition
R = physical_constants['molar gas constant'][0]
F = physical_constants['Faraday constant'][0]
kB = physical_constants['Boltzmann constant'][0]
h = physical_constants['Planck constant'][0]
N_A = physical_constants['Avogadro constant'][0]
eq = physical_constants['elementary charge'][0]
E0 = 8.854187e-12  # Farad/m

# Neighbors' position definition
neighbors = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
             (-1, 0, 0), (0, -1, 0), (0, 0, -1)]

face_neighbors = {(1, 0, 0), (0, 1, 0), (0, 0, 1),
                  (-1, 0, 0), (0, -1, 0), (0, 0, -1)}

event_code = {
    (1, 0, 0): 1,
    (0, 1, 0): 2,
    (0, 0, 1): 3,
    (-1, 0, 0): 4,
    (0, -1, 0): 5,
    (0, 0, -1): 6,
    'MV charge': 7,
    'MV discharge': 8,
    'TEMPO charge': 9,
    'TEMPO discharge': 10,
    'Formation of dimers': 11,
    'Disproportionation': 12,
    'MV Adsoption': 13,
    'MV Desorption': 14
}

event_decode = dict((value, key) for key, value in event_code.items())  # value and key change place

def sinh(X):
    return float((exp(X)-exp(-X)) / 2)

def cosh(X):
    return float((exp(X)+exp(-X)) / 2)

@lru_cache()
def calc_diff_coef(viscosity, temperature, radius):
    k_diff = kB * temperature / (6 * pi * viscosity * radius)
    return k_diff

class SubParticle(object):
    '''
    for multi-subparticles particle, define subparticle object.
    x,y,z --> subparticle position
    parent --> parent position
    particle_charge --> parent particle-charge
    '''
    def __init__(self, x, y, z, parent, particle_charge):
        self.name = (x, y, z)
        self.set_position((x, y, z))
        self.parent = parent
        self.particle_charge = particle_charge
        self.neighbors = {(x + i, y + j, z + k,) for i, j, k in neighbors
                          if x + i >= 0 and y + j >= 0 and z + k >= 0}

    def get_face_neighbors(self):
        x, y, z = self.position
        for o, p, q in face_neighbors:
            yield x + o, y + p, z + q

    def set_position(self, position):
        self.x, self.y, self.z = position
        self.position = position
        self.name = position

    def move(self, direction):
        o, p, q = direction
        new_position = (self.x + o, self.y + p, self.z + q)
        self.set_position(new_position)
        return self.neighbors

    def __bool__(self):
        return True

    def __hash__(self):
        return hash(self.name)

    # Output .xyz file content
    def __str__(self):
        return f'{self.particle_charge:4d} {self.x:2d} {self.y:2d} {self.z:2d}'

    def __repr__(self):
        return f'SubParticle(position={self.name}Parent{self.parent})'


class Particle(object):
    def __init__(self, index, position, shape, radius, particle_charge, elec_field):
        self.index = index
        x, y, z = position
        self.set_position(position)
        self.particle_charge = particle_charge
        self.radius = radius
        self.elec_field = elec_field

        # Call the SubParticle class for each SubParticle
        shape = shape.lower()
        if shape == "cubic":
            children = {(x, y, z): SubParticle(x, y, z, self, self.particle_charge)}

        elif shape == "sphere":
            children = {(x + i, y + j, z + k): SubParticle(x + i, y + j, z + k, self, particle_charge)
                        for i, j, k in {(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                                        (-1, 0, 0), (0, -1, 0), (0, 0, -1)}}

        elif shape == "double units":
            sub_p_index = randint(0, 5)
            children = {(x + i, y + j, z + k): SubParticle(x + i, y + j, z + k, self, particle_charge)
                        for i, j, k in {(0, 0, 0), neighbors[sub_p_index]}}

        else:
            raise ValueError(f"Shape {shape} not supported.")

        self.sub_particles = children

    def set_position(self, position):
        self.x, self.y, self.z = position
        self.position = position
        return self.position

    def move(self, direction):
        """
        compute a new position and update it for self and all sub particles
        :param direction:
        :return:
        """
        o, p, q = direction
        x, y, z = self.position
        new_position = x + o, y + p, z + q
        self.set_position(new_position)
        self.move_sub_particles(direction)
        return new_position

    def move_sub_particles(self, direction):
        new_sub_particles = {}
        for sub_p in self.sub_particles.values():
            sub_p.move(direction)
            # Generate a new dictionary with new sub_p positions
            new_sub_particles[sub_p.position] = sub_p
        self.sub_particles = new_sub_particles

    def get_face_neighbors(self):
        for sub_p in self:
            for neigh in sub_p.get_face_neighbors():
                yield neigh

    def get_diffusion_events(self, grid, Z):
        for o, p, q in neighbors:
            new_position = []
            for x, y, z in self.sub_particles:
                self.direction = (o, p, q)
                new_pos = (x + o, y + p, z + q)
                if z + q <= Z:
                    if new_pos not in self.sub_particles:
                        sub_p = grid.get(new_pos)
                        if sub_p is None:
                            new_position.append(new_pos)
                            self.calc_reaction_units()
                            yield (o, p, q), new_position, self.reaction_units


    def calc_reaction_units(self):
        """
        count the number of reaction surface in the particle based on diffusion directions.
        For example, a particle of two units (0,1,0) and (0,1,1) with the direction of (0,1,0)
        will have two reaction surfaces. On the other hand, if this the direction is (0,0,1),
        there will be only one reaction surface
        :return:
        """
        count = set()
        for i in range(0, 3):
            if self.direction[i] != 0:
                for particle in self.sub_particles:
                    count.add(particle[i])
                self.reaction_units = len(self.sub_particles) + 1 - len(count)

    def __len__(self):
        return len(self.sub_particles)

    def __iter__(self):
        for sub_p in self.sub_particles.values():
            yield sub_p


class MV(Particle):
    def __init__(self, index, position, shape, radius, particle_charge, elec_field):
        super().__init__(index, position, shape, radius, particle_charge, elec_field)

    def __repr__(self):
        return f'MV(index={self.index}, position{self.position})'

class Cl_ion(Particle):
    def __init__(self, index, position, shape, radius, particle_charge, elec_field):
        super().__init__(index, position, shape, radius, particle_charge, elec_field)

    def __repr__(self):
        return f'Cl_ion(index={self.index}, position{self.position})'


class TEMPO(Particle):
    def __init__(self, index, position, shape, radius, particle_charge, elec_field):
        super().__init__(index, position, shape, radius, particle_charge, elec_field)

    def __repr__(self):
        return f'TEMPO(index={self.index}, position={self.position})'

class Board():
    def __init__(self, run_info, board_info, operation_conditions, electrolyte_info,
                 electrochemistry_info, MV_info, *args, **kwargs):

        self.simu_condiiton_setting(**run_info)
        self.set_board_info(board_info, operation_conditions, electrolyte_info)
        self.set_operation_conditions(operation_conditions)
        self.set_electrolyte_info(electrolyte_info)
        self.set_MV_info(MV_info)
        self.set_chloride_info(MV_info, electrolyte_info)
        self.set_electrochemistry_info(electrochemistry_info)
        self.set_water_adsorption_info()
        self.define_A_coefficient_matrix()
        self.grid = {}       # Grid is a dictionary，key is the 3D location，value is the object of each SubParticles
        self.initialize_particles()
        self.time = 0

    def simu_condiiton_setting(self, name, **kwargs):
        self.file_xyz = name + '.xyz'
        self.file_pot = name + '.out'
        self.file_psd = name + '.txt'

        self.iteration = kwargs['max_iterations']
        self.frequency = kwargs['frequency']
        self.cut_off_time = kwargs['time']

    def set_board_info(self, board_info, operation_condition, electrolyte_info):
        X, Y, Z = board_info['board size']
        self.X = X
        self.Y = Y
        self.n_dl = Z                        # number of debye layer to consider for the simulation area
        self.unit_scale = board_info['unit scale']
        self.init_conc_mv = electrolyte_info['mv initial concentration']
        self.init_conc_NaCl = electrolyte_info['solution initial concentration']
        self.mv_init_soc = electrolyte_info['mv initial soc'] / 100
        self.mv_permittivity = electrolyte_info['mv solution permittivity']
        self.temperature = operation_condition['temperature']

        self.debye_length = (kB * self.temperature * E0 * self.mv_permittivity/
                             ((self.mv_init_soc * self.init_conc_mv * eq ** 2 * N_A) +
                              ((1-self.mv_init_soc) * self.init_conc_mv * 4 * eq ** 2 * N_A) +
                              ((2 * self.init_conc_mv - self.mv_init_soc * self.init_conc_mv + self.init_conc_NaCl) * eq ** 2 * N_A)
                              + (self.init_conc_NaCl * eq ** 2 * N_A))) ** 0.5
        print('debye length=', self.debye_length)

        self.Z = round(self.n_dl * self.debye_length / self.unit_scale) + 1
        self.board_size = (X, Y, self.Z)

        self.total_soc = 0
        self.total_volume_units = self.X * self.Y * self.Z
        self.unit_volume_cube = self.unit_scale ** 3  # m^3
        self.unit_volume_sphere = (4 * pi * (self.unit_scale / 2) ** 3) / 3  # m^3
        self.shape = board_info['unit shape']
        self.event_type = 'Nah'
        self.dt = 0

        if self.shape == 'cubic':
            self.total_volume = self.total_volume_units * self.unit_volume_cube  # m^3
        elif self.shape == 'sphere':
            self.total_volume = self.total_volume_units * self.unit_volume_sphere  # m^3
        else:
            raise ValueError(f"Shape {self.shape} not supported")

    def set_operation_conditions(self, operation_condition):
        self.exp_electrode_surface = 5e-4       #m^2
        self.exp_contact_surface = 22400 * 2e-3 * self.exp_electrode_surface
        self.j_input_dis = operation_condition['input current density'] * self.exp_electrode_surface / self.exp_contact_surface
        self.j_input = 0
        print('real J input', self.j_input)

    def set_electrolyte_info(self, electrolyte_info):
        self.init_viscosity_MV = electrolyte_info['mv initial viscosity']
        self.conc_mv_discharged = 1e-9
        self.conc_mv_charged = self.init_conc_mv
        self.conc_mv_degradate = 0

    def set_MV_info(self, mv_info):
        self.mv_molar_mass = mv_info['mv molar mass']
        self.mv_charged_radius = mv_info['mv gyration radius before discharging']
        self.mv_discharged_radius = mv_info['mv gyration radius after discharging']
        self.mv_standard_potential = mv_info['mv standard electrode potential']
        self.mv_info = {'shape': mv_info['shape'],
                        'radius': mv_info['mv gyration radius before discharging'],
                        'particle_charge': mv_info['mv charged valence'],
                        'elec_field': 0}
        self.mv_d_info = {'shape': mv_info['shape'],
                          'radius': mv_info['mv gyration radius after discharging'],
                          'particle_charge': mv_info['mv discharged valence'],
                          'elec_field': 0}
        self.n_particle = int(self.conc_mv_charged * N_A * self.total_volume * 1000)
        self.n_mv_nd = int(self.n_particle * self.mv_init_soc)
        print('number of mv', self.n_mv_nd)
        self.n_mv_d = self.n_particle - self.n_mv_nd

        self.n_mv_deg = 0
        self.d_number_MV = 0
        self.d_number_MV_deg = 0
        self.d_number_Cl = 0
        self.capacitance = 0

    def set_chloride_info(self, mv_info, cl_info):
        self.radius_Cl = cl_info['cl radius'] # [m]
        self.k_diff_Cl = kB * self.temperature / (6 * pi * self.init_viscosity_MV * self.radius_Cl)
        self.n_cl = mv_info['mv discharged valence'] * self.n_mv_d + self.n_mv_nd
        self.cl_info = {'shape': mv_info['shape'],
                        'radius': cl_info['cl radius'],
                        'particle_charge': cl_info['cl charge'],
                        'elec_field': 0}
        if self.n_particle + self.n_cl >= self.X * self.Y * self.Z:
            raise ValueError('Volume limit')

    def set_electrochemistry_info(self, electrochemistry_info):
        self.inner_layer_permittivity = electrochemistry_info['compact layer permittivity']
        self.oxi_lambda_in = electrochemistry_info['oxidation reorganization energy']
        self.red_lambda_in = electrochemistry_info['reduction reorganization energy']
        self.tunneling_distance = electrochemistry_info['tunneling distance']

        self.tunneling_layers = round((self.tunneling_distance + 0.5 * self.unit_scale) / self.unit_scale)
        self.tunneling_limit = self.Z - self.tunneling_layers

        self.simu_box_limit = 1
        self.calc_lambda_tot()                             # Calculate the total reorganisation energy
        self.compact_layer_d = electrochemistry_info['compact layer thickness']
        self.compact_layer_limit = self.Z - round(self.compact_layer_d / self.unit_scale)
        self.electrode_potential = 0
        self.surface_area = self.unit_scale ** 2
        self.E = self.mv_standard_potential
        self.charge_density = -0
        self.init_charge_density = - 0
        self.potential_drop = 0
        self.j_far = 0
        self.phi_inter = 0
        self.total_discharge = 0
        self.electrode_surface = 5e3 * self.X * self.Y * self.unit_scale ** 2

        # generate the initial dictionary for discharge rate based on tunnelling distance
        self.discharge_rate = {i : 0 for i in range(1, self.tunneling_layers+1)}
        self.charge_rate = self.discharge_rate

        self.deg_reo_nrg = electrochemistry_info['degradation energy']*F
        self.delta_E = 0
        self.ext_box_n_mv_nd = int(self.simu_box_limit * self.unit_volume_cube * self.X * self.Y * self.init_conc_mv * \
                                     self.mv_init_soc * N_A * 1000)
        self.ext_box_n_mv_d = int(self.simu_box_limit * self.unit_volume_cube * self.X * self.Y * self.init_conc_mv * \
                                    (1 - self.mv_init_soc) * N_A * 1000)
        self.ext_box_n_cl = 2 * self.ext_box_n_mv_d + self.ext_box_n_mv_nd
        self.cl_capacitance = self.inner_layer_permittivity * E0 / (4 * pi * self.compact_layer_d)
        print(self.ext_box_n_mv_nd, self.ext_box_n_mv_d, self.ext_box_n_cl)
        print('debye_length', self.debye_length, 'z axis', self.Z, 'tunneling_limit', self.tunneling_limit,
              'compact_limit', self.compact_layer_limit, 'simulation box limit', self.simu_box_limit)

    def calc_lambda_tot(self):
        # calculate the total reorganization energy
        self.oxy_lambda_tot = {1: self.oxi_lambda_in*eq*N_A}
        self.red_lambda_tot = {1: self.red_lambda_in*eq*N_A}
        R = [1.5 * self.unit_scale, 2.5 * self.unit_scale, 3.5 * self.unit_scale]
        R = np.array(R)
        lambda_tot_oxy = eq * N_A * (self.oxi_lambda_in + (eq / (8 * pi) * ((1 / self.inner_layer_permittivity / E0) - (1 / self.mv_permittivity / E0))
                                                           * ((1 / self.mv_charged_radius) - (0.5 / R))))
        lambda_tot_red = eq * N_A * (self.red_lambda_in + (eq / (8 * pi) * ((1 / self.inner_layer_permittivity / E0) - (1 / self.mv_permittivity / E0))
                                                           * ((1 / self.mv_discharged_radius) - (0.5 / R))))
        i = 1
        while i <= 3:
            self.oxy_lambda_tot[i+1] = lambda_tot_oxy[i-1]
            self.red_lambda_tot[i+1] = lambda_tot_red[i-1]
            i += 1
        return self.oxy_lambda_tot, self.red_lambda_tot

    def set_water_adsorption_info(self):
        self.adsorption_energy_water = 1000  # J/mol
        self.water_mole_thicknes = 2e-10  # m
        self.water_dipolar_moment = 0.617e-29  # C*m
        self.water_activity_coefficient = 55000  # mol/m^3
        self.water_n_max = 1 / self.water_mole_thicknes ** 2
        self.water_theta = 0.95
        self.water_a = 2 * self.water_theta * exp(-self.adsorption_energy_water / (R * self.temperature))
        self.water_b = self.water_dipolar_moment / (E0 * self.inner_layer_permittivity * kB * self.temperature)
        self.water_c = 1.202 * self.water_dipolar_moment ** 2 / (2 * pi * kB * self.temperature * self.mv_permittivity
                                                                 * E0 * self.water_mole_thicknes ** 3)
        self.water_X = 0
        self.theta_free = 0
        self.adsorbed_particles = []

    def calc_conc_MV(self):
        self.conc_mv_discharged = (self.n_mv_d / (1000 * N_A * self.total_volume))
        self.conc_mv_degradate = (self.n_mv_deg / (1000 * N_A * self.total_volume))
        self.conc_mv_charged = (self.n_mv_nd / (1000 * N_A * self.total_volume))

    def get_total_soc(self):
        self.total_soc = 0
        for particle in self.particles:
            self.total_soc += particle.particle_charge
        return self.total_soc

    def add_particles(self, particles):
        self.grid.update(particles)

    def initialize_particles(self):
        self.mv_nd_init = []
        self.mv_d_init = []
        self.cl_ion_init = []
        for i in range(self.n_mv_nd):
            while True:
                x = randint(1, self.X)
                y = randint(1, self.Y)
                z = randint(1, self.Z)
                tmp_p = MV(i, (x, y, z), **self.mv_info)
                if not any(self.grid.get(position, False)
                           for position in tmp_p.sub_particles):
                    self.mv_nd_init.append(tmp_p)
                    self.add_particles(tmp_p.sub_particles)
                    break

        for i in range(self.n_mv_nd, self.n_mv_d + self.n_mv_nd):
            while True:
                x = randint(1, self.X)
                y = randint(1, self.Y)
                z = randint(1, self.Z)
                tmp_p = MV(i, (x, y, z), **self.mv_d_info)
                if not any(self.grid.get(position, False)
                           for position in tmp_p.sub_particles):
                    self.mv_d_init.append(tmp_p)
                    self.add_particles(tmp_p.sub_particles)
                    break

        for i in range(self.n_mv_d + self.n_mv_nd, self.n_mv_d + self.n_mv_nd + self.n_cl):
            while True:
                x = randint(1, self.X)
                y = randint(1, self.Y)
                z = randint(1, self.Z)
                tmp_p = Cl_ion(i, (x, y, z), **self.cl_info)
                if not any(self.grid.get(position, False)
                           for position in tmp_p.sub_particles):
                    self.cl_ion_init.append(tmp_p)
                    self.add_particles(tmp_p.sub_particles)
                    break

        self.particles = self.mv_nd_init + self.mv_d_init + self.cl_ion_init
        self.calc_electrode_surface_concentration()
        self.calc_big_gamma_water()
        self.calc_reaction_rate()
        self.calc_dimers_form_rate()
        self.define_b_matrix()
        self.solve_poisson()
        self.update_elec_field()

    def add_Cl(self, Z_axis):
        while True:
            i = len(self.particles)
            x = randint(1, self.X)
            y = randint(1, self.Y)
            z = Z_axis
            tmp_p = Cl_ion(i, (x, y, z), **self.cl_info)
            if not any(self.grid.get(position, False)
                       for position in tmp_p.sub_particles):
                self.particles.append(tmp_p)
                self.add_particles(tmp_p.sub_particles)
                break

    def del_Cl(self, number):
        chloride_particle = []
        for particle in self.particles:
            if particle.particle_charge == -1:
                chloride_particle.append(particle)
        sup_cl_ion = np.random.choice(chloride_particle, size=(number), replace=False)
        for cl_ion in sup_cl_ion:
            del self.grid[cl_ion.position]
            self.particles.remove(cl_ion)
        self.get_total_soc()

    def add_MV_nd(self, Z_axis):
        while True:
            i = len(self.particles)
            x = randint(1, self.X)
            y = randint(1, self.Y)
            z = Z_axis
            tmp_p = MV(i, (x, y, z), **self.mv_info)
            if not any(self.grid.get(position, False)
                       for position in tmp_p.sub_particles):
                self.particles.append(tmp_p)
                self.add_particles(tmp_p.sub_particles)
                break

    def add_MV_d(self, Z_axis):
        while True:
            i = len(self.particles)
            x = randint(1, self.X)
            y = randint(1, self.Y)
            z = Z_axis
            tmp_p = MV(i, (x, y, z), **self.mv_d_info)
            if not any(self.grid.get(position, False)
                       for position in tmp_p.sub_particles):
                self.particles.append(tmp_p)
                self.add_particles(tmp_p.sub_particles)
                break

    def calc_water_X1(self):
        self.water_X1 = self.water_b * self.charge_density + (self.water_a * self.water_c * sinh(-self.water_X)
                                                              / (1 + self.water_a * cosh(self.water_X)))

    def calc_water_X(self):
        self.water_X = 0
        self.calc_water_X1()
        while abs(self.water_X - self.water_X1) >= 1e-4:
            if self.charge_density > 0:
                self.water_X += 1e-4
                self.calc_water_X1()
            elif self.charge_density < 0:
                self.water_X -= 1e-4
                self.calc_water_X1()
            elif self.charge_density == 0:
                self.water_X = 0
        return self.water_X

    def calc_reaction_rate(self):
        T = self.temperature
        self.potential_drop = (self.charge_density * self.compact_layer_d / (E0 * self.inner_layer_permittivity)) #\
                              # + self.big_gamma / (self.mv_permittivity * E0)
        i = 1
        while i < 5:
            Ea_dc = self.oxy_lambda_tot.get(i)
            Ea_c = self.red_lambda_tot.get(i)
            T = self.temperature
            self.discharge_rate[i] = ((kB * T) / h) * exp((- Ea_dc + 0.5 * F * self.potential_drop) / (R * T))
            self.charge_rate[i] = 100*((kB * T) / h) * exp((- Ea_c - F * self.potential_drop) / (R * T))
            i += 1
        return self.discharge_rate, self.charge_rate

    def get_reaction_rate(self, particle_z, particle_charge):
        key = int(self.Z - particle_z + 1)
        if particle_charge == 1:
            return self.discharge_rate.get(key)
        elif particle_charge == 2:
            return self.charge_rate.get(key)

    def calc_theta_free_water(self):
        self.theta_free = 1 / (1 + self.water_a * cosh(self.water_X))
        return self.theta_free

    def calc_big_gamma_water(self):
        self.calc_water_X()
        self.calc_theta_free_water()
        self.big_gamma = self.water_a * self.water_dipolar_moment * sinh(-self.water_X) * self.theta_free \
                         / (self.water_mole_thicknes ** 2)
        return self.big_gamma

    def calc_dimers_form_rate(self):
        self.dimers_rate = ((kB * self.temperature) / h) * exp((-self.deg_reo_nrg) / (R * self.temperature))
        return self.dimers_rate

    def dimers_formation(self, particle1, particle2):
        particle1.particle_charge = 2
        particle2.particle_charge = 0
        self.get_total_soc()
        self.update_children_soc(particle1)
        self.update_children_soc(particle2)
        return

    def calc_reaction_area(self, reaction_units):
        self.reaction_area = reaction_units * self.unit_scale ** 2
        return self.reaction_area

    def add_ele_field_impact(self, direction, rate, ele_field):
        if direction[2] > 0 and rate != 0:
            diff_rate = rate * ele_field
        elif direction[2] < 0 and rate != 0:
            diff_rate = rate / ele_field
        else:
            diff_rate = rate
        return diff_rate

    def form_of_dimers(self):
        pass

    def get_diff_rate(self, particle, reaction_units, direction):
        self.calc_reaction_area(reaction_units)
        rate = calc_diff_coef(self.init_viscosity_MV, self.temperature, particle.radius) / self.surface_area
        ele_field = exp(particle.elec_field * eq * particle.radius * particle.particle_charge
                        / (kB * self.temperature))

        if ele_field > 100:
            while ele_field > 10:
                ele_field /= 10

        if ele_field < 0.01:
            while ele_field < 0.1:
                ele_field *= 10

        diff_rate = self.add_ele_field_impact(direction, rate, ele_field)
        return diff_rate

    def in_bounds(self, pos):
        """
        Verify if the particle position is inside the simulation box
        :param pos:
        :return: particle inside the simulation box
        """
        return all(1 <= c <= l for c, l in zip(pos, self.board_size))

    def is_free(self, pos):
        """
        Grid is a dictionary
        ruturn empty sub_p
        :param pos:
        :return:
        """
        if not self.in_bounds(pos):
            return False
        sub_p = self.grid.get(pos)
        return sub_p is None

    def is_not_free(self, pos):
        if not self.in_bounds(pos):
            return False
        sub_p = self.grid.get(pos)
        return sub_p is not None

    def compile_rate_list(self):
        rate_list = []
        events = []

        for particle in self.particles:
            # Reaction event
            if particle.z > self.tunneling_limit and particle.particle_charge == 1:
                pass
                # reaction_rate = self.get_reaction_rate(particle.z, particle.particle_charge)
                # self.reaction_type = 'MV discharge'
                # rate_list.append(reaction_rate)
                # events.append((particle, event_code[self.reaction_type]))

            elif particle.z > self.tunneling_limit and particle.particle_charge == 2:
                reaction_rate = self.get_reaction_rate(particle.z, particle.particle_charge)
                self.reaction_type = 'MV charge'
                rate_list.append(reaction_rate)
                events.append((particle, event_code[self.reaction_type]))

            # Diffusion event
            for direction, new_position, reaction_units in particle.get_diffusion_events(self.grid, self.Z):
                diff_rate = self.get_diff_rate(particle, reaction_units, direction)
                for pos in new_position:
                    if pos[2] <= self.Z:
                        rate_list.append(diff_rate)
                        events.append((particle, event_code[direction]))

                # Degradation event
                if particle.particle_charge == 1:
                    for pos in new_position:
                        if self.is_not_free(pos):
                            for p in self.particles:
                                if p.position == pos and p.particle_charge == 1:
                                    # self.calc_dimers_form_rate()
                                    rate_list.append(self.dimers_rate)
                                    events.append(((particle, p), 11))

        self.kmc_rates = rate_list
        self.kmc_events = events

    def compile_rate_list_FIS(self):
        rate_list = []
        events = []

        for particle in self.particles:
            # Diffusion event
            for direction, new_position, reaction_units in particle.get_diffusion_events(self.grid, self.Z):
                diff_rate = self.get_diff_rate(particle, reaction_units, direction)
                for pos in new_position:
                    if pos[2] <= self.Z:
                        rate_list.append(diff_rate)
                        events.append((particle, event_code[direction]))
                        print(rate_list)

        self.kmc_rates = rate_list
        self.kmc_events = events

    def update_children(self, particle):
        for position, sub_p in particle.sub_particles.items():
            self.grid[position] = sub_p

    def update_children_soc(self, particle):
        for position, sub_p in particle.sub_particles.items():
            sub_p.particle_charge = particle.particle_charge

    def discharge(self, particle):
        particle.particle_charge = 2
        self.get_total_soc()
        self.update_children_soc(particle)
        return

    def charge(self, particle):
        particle.particle_charge = 1
        self.get_total_soc()
        self.update_children_soc(particle)
        return

    def move(self, particle, direction):
        old_position = particle.sub_particles.keys()
        o, p, q = direction
        x, y, z = particle.position
        new_position = (x+o, y+p, z+q)
        new_pos = self.grid.get(new_position)
        if new_pos is None:
            particle.move(direction)
            for pos in old_position:
                del self.grid[pos]
            if not self.in_bounds(new_position):
                particle_soc = particle.particle_charge
                self.particles.remove(particle)
                if new_position[2] < 1 or new_position[2] > self.Z:
                    # if the molecule evacuate from edl, don’t care
                    print('going to delet', particle)
                    if particle_soc == 1:
                        self.n_mv_nd -= 1
                    elif particle_soc == 2:
                        self.n_mv_d -= 1
                    elif particle_soc == -1:
                        self.n_cl -= 1
                else:
                    #if the molecule evacuate from x, y direction, periodic BC
                    if particle_soc == 1:
                        self.add_MV_nd(z)
                    elif particle_soc == 2:
                        self.add_MV_d(z)
                    elif particle_soc == -1:
                        self.add_Cl(z)
            else:
                self.update_children(particle)
        else:
            pass
        return

    def move_init(self, particle, direction):
        old_position = particle.sub_particles.keys()
        o, p, q = direction
        x, y, z = particle.position
        new_position = (x + o, y + p, z + q)
        new_pos = self.grid.get(new_position)
        if new_pos is None:
            particle.move(direction)
            for pos in old_position:
                del self.grid[pos]
            if not self.in_bounds(new_position):
                particle_soc = particle.particle_charge
                self.particles.remove(particle)
                if particle_soc == 1:
                    self.add_MV_nd(z)
                elif particle_soc == 2:
                    self.add_MV_d(z)
                elif particle_soc == -1:
                    self.add_Cl(z)
            else:
                self.update_children(particle)
        else:
            pass
        return

    def check_ETB_concentration(self):
        '''

        :return:
        '''
        n_mv_nd = []
        n_mv_d = []
        n_cl = []
        for particle in self.particles:
            if particle.position[2] <= self.simu_box_limit:
                if particle.particle_charge == 1:
                    n_mv_nd.append(particle)
                elif particle.particle_charge == 2:
                    n_mv_d.append(particle)
                elif particle.particle_charge == -1:
                    n_cl.append(particle)

        # adjust MV2+ concentration in the extend box
        while len(n_mv_d) < self.ext_box_n_mv_d:
            z = randint(1, self.simu_box_limit)
            self.add_MV_d(z)
            n_mv_d.append(1)
        while len(n_mv_d) > self.ext_box_n_mv_d:
            del_mv_ion = np.random.choice(n_mv_d, size=(len(n_mv_d)-self.ext_box_n_mv_d), replace=False)
            for mv in del_mv_ion:
                del self.grid[mv.position]
                self.particles.remove(mv)
                n_mv_d.remove(mv)
            self.get_total_soc()

        # adjust MV+ concentration in the extend box
        while len(n_mv_nd) < self.ext_box_n_mv_nd:
            z = randint(1, self.simu_box_limit)
            self.add_MV_nd(z)
            n_mv_nd.append(1)
        while len(n_mv_nd) > self.ext_box_n_mv_nd:
            del_mv_ion = np.random.choice(n_mv_nd, size=(len(n_mv_nd) - self.ext_box_n_mv_nd), replace=False)
            for mv in del_mv_ion:
                del self.grid[mv.position]
                self.particles.remove(mv)
                n_mv_nd.remove(mv)
            self.get_total_soc()

        # adjust Cl- concentration in the bulk
        while len(n_cl) < self.ext_box_n_cl:
            z = randint(1, self.simu_box_limit)
            self.add_Cl(z)
            n_cl.append(1)
        while len(n_mv_nd) > self.ext_box_n_mv_nd:
            del_cl_ion = np.random.choice(n_cl, size=(len(n_cl) - self.ext_box_n_cl), replace=False)
            for cl in del_cl_ion:
                del self.grid[cl.position]
                self.particles.remove(cl)
                n_cl.remove(cl)
            self.get_total_soc()
        return

    def execute_event(self, particle, eventcode):
        if isinstance(event_decode[eventcode], tuple):
            self.move(particle, event_decode[eventcode])
            self.del_q = 0
            self.event_type = 'Diffusion'

        elif eventcode == 8:
            self.discharge(particle)
            self.del_q = - eq
            self.n_mv_d += 1
            self.event_type = 'MV_Discharge'

        elif eventcode == 7:
            self.charge(particle)
            self.del_q = eq
            self.n_mv_d -= 1
            self.event_type = 'MV_Charge'

        elif eventcode == 11:
            main_particle = particle[0]
            sec_particle = particle[1]
            self.dimers_formation(main_particle, sec_particle)
            self.del_q = 0
            self.d_number_MV += 1
            self.d_number_MV_deg += 1
            self.event_type = 'Degradation'

        self.check_ETB_concentration()

    def execute_event_IP(self, particle, eventcode):
        if isinstance(event_decode[eventcode], tuple):
            self.move_init(particle, event_decode[eventcode])
            self.del_q = 0
            self.d_number_MV += 0
            self.event_type = 'Diffusion'

        self.check_ETB_concentration()

    def vssm_sonar(self):
        self.compile_rate_list()
        rates = np.array(self.kmc_rates, dtype=float)
        self.K_total = rates.sum()
        probabilities = rates / self.K_total
        n_events = len(rates)
        if not n_events:
            print('no more event available')
            raise ValueError("empty rate list")
        else:
            i = np.random.choice(range(n_events), p=probabilities)

            particle = self.kmc_events[i][0]
            eventcode = self.kmc_events[i][1]

            self.execute_event(particle, eventcode)

            r = random()
            self.dt = -log(r) / self.K_total
            self.time += self.dt
            self.total_discharge += self.del_q

        self.j_far = self.del_q / (self.dt * self.electrode_surface)

        self.charge_density = self.charge_density + (self.j_far - self.j_input) * (self.dt)

        self.calc_water_X()
        self.calc_big_gamma_water()
        self.calc_reaction_rate()
        self.calc_potential()

    def define_b_matrix(self):

        self.auxgrid = np.zeros((self.NX,self.NY,self.NZ))
        #Initialize rho in Poisson equation
        self.charge_array = np.zeros(self.Ntot)

        for particle in self.particles:
            if particle.z <= self.compact_layer_limit and particle.z >= self.simu_box_limit: # this <= depends on the sign in initialize_particles function
                #Translate particle position to auxiliar mesh
                xaux = int(0.5*self.N)*(2*particle.x - 1)
                yaux = int(0.5*self.N)*(2*particle.y - 1)
                zaux = int(0.5*self.N)*(2*(particle.z - self.simu_box_limit) - 1)
                self.auxgrid[xaux,yaux,zaux] = -(1.0/self.h**3)*eq*particle.particle_charge/(self.mv_permittivity*E0)

        p = 0
        for i in range(self.NX):
            for j in range(self.NY):
                for k in range(self.NZ):
                    self.charge_array[p] = self.auxgrid[i,j,k]
                    p += 1

        #Set Neuman BC
        for i in self.nbc:
            self.charge_array[i] = self.charge_array[i] - 2.0*self.charge_density/(self.mv_permittivity*E0*self.h)

        self.charge_array = self.charge_array*self.h**2
        #Set Dirichlet BC
        for i in self.dbc:
            self.charge_array[i] = 0.0

        return self.charge_array

    def solve_poisson(self):
        #Solve Ax=b linear set of equations, with obtained A and b        
        self.phi = spsolve(self.coef_mat, self.charge_array)

        aux = np.zeros((self.NX,self.NY,self.NZ))
        p = 0
        for i in range(self.NX):
            for j in range(self.NY):
                for k in range(self.NZ):
                    aux[i,j,k] = self.phi[p]
                    p += 1

        self.phi = aux

        self.elec_field = np.gradient(-1.0*self.phi,self.h,edge_order=2)
        self.reaction_plane_E = np.mean(self.elec_field[2][:,:,self.NZ-1])
        return self.phi, self.elec_field

    def define_A_coefficient_matrix(self):

        self.N = 2
        self.NX = self.N*(self.X)+1
        self.NY = self.N*(self.Y)+1
        self.NZ = self.N * (self.compact_layer_limit - self.simu_box_limit + 1) + 1
        self.phi = np.zeros((self.NX,self.NY,self.NZ))

        #Define laplace operator in 3D as a square matrix
        # \lambda phi|_ijk =   (u_i-1jk + u_i+1jk - 2*u_ijk)/dx^2 +
        #                    + (u_ij-1k + u_ij+1k - 2*u_ijk)/dy^2 +
        #                    + (u_ijk-1 + u_ijk+1 - 2*u_ijk)/dz^2

        self.h = self.unit_scale/self.N

        # Size of matrix 
        self.Ntot = (self.NX)*(self.NY)*(self.NZ)

        # Initialize arrays to store general row and column indeces
        self.rows = np.empty(0,dtype=int)
        self.columns = np.empty(0,dtype=int)

        # Initialize array to store position of DIRICHLET BC
        self.dbc = np.empty(0,dtype=int)

        # Initialize array to store position of NEUMAN BC
        self.nbc = np.empty(0,dtype=int)

        #Initialize array to store the data
        self.coefs = np.empty(0,dtype=float)

        # Indexing of flattened array works as p = NZ*NY*i + NZ*j + k
        p = 0
        for i in range(self.NX):
            for j in range(self.NY):
                for k in range(self.NZ):
                    #Define 7 point stencil                  
                    pxu = p + self.NZ*self.NY
                    pxd = p - self.NZ*self.NY
                    pyu = p + self.NZ
                    pyd = p - self.NZ
                    pzu = p + 1
                    pzd = p - 1

                    if k == 0: #Nodes with Dirichlet boundary conditions in z=0
                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,p)
                        self.coefs = np.append(self.coefs,1.0)

                        self.dbc = np.append(self.dbc,p)

                    elif i==0 and j==0 and k!=0 and k!=(self.NZ-1): #Nodes with PBC in the EDGE x=0 y=0

                        #Redefine stencil points that fall outside of domain such that U-11 -> U(N-1)1
                        pxd = p + self.NZ*self.NY*(self.NX-2)
                        pyd = p + self.NZ*(self.NY-2)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,p)
                        self.coefs = np.append(self.coefs,-6.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pzu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pzd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyd)
                        self.coefs = np.append(self.coefs,1.0)

                    elif i==(self.NX-1) and j==0 and k!=0 and k!=(self.NZ-1): #Nodes with PBC in the EDGE x=L y=0

                        #Redefine stencil points that fall outside of domain
                        pxu = p - self.NZ*self.NY*(self.NX-2)
                        pyd = p + self.NZ*(self.NY-2)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,p)
                        self.coefs = np.append(self.coefs,-6.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pzu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pzd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyd)
                        self.coefs = np.append(self.coefs,1.0)

                    elif i==0 and j==(self.NY-1) and k!=0 and k!=(self.NZ-1): #Nodes with PBC in the EDGE x=0 y=L

                        #Redefine stencil points that fall outside of domain
                        pxd = p + self.NZ*self.NY*(self.NX-2)
                        pyu = p - self.NZ*(self.NY-2)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,p)
                        self.coefs = np.append(self.coefs,-6.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pzu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pzd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyd)
                        self.coefs = np.append(self.coefs,1.0)

                    elif i==(self.NX-1) and j==(self.NY-1) and k!=0 and k!=(self.NZ-1): #Nodes with PBC in the EDGE x=L y=L

                        #Redefine stencil points that fall outside of domain
                        pxu = p - self.NZ*self.NY*(self.NX-2)
                        pyu = p - self.NZ*(self.NY-2)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,p)
                        self.coefs = np.append(self.coefs,-6.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pzu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pzd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyd)
                        self.coefs = np.append(self.coefs,1.0)

                    elif i==0 and j!=0 and j!=(self.NY-1) and k!=0 and k!=(self.NZ-1): #Nodes with PBC in the FACE x=0

                        #Redefine stencil points that fall outside of domain
                        pxd = p + self.NZ*self.NY*(self.NX-2)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,p)
                        self.coefs = np.append(self.coefs,-6.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pzu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pzd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyd)
                        self.coefs = np.append(self.coefs,1.0)

                    elif i==(self.NX-1) and j!=0 and j!=(self.NY-1) and k!=0 and k!=(self.NZ-1): #Nodes with PBC in the FACE x=L

                        #Redefine stencil points that fall outside of domain
                        pxu = p - self.NZ*self.NY*(self.NX-2)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,p)
                        self.coefs = np.append(self.coefs,-6.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pzu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pzd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyd)
                        self.coefs = np.append(self.coefs,1.0)

                    elif i!=0 and i!=(self.NX-1) and j==0 and k!=0 and k!=(self.NZ-1): #Nodes with PBC in the FACE y=0

                        #Redefine stencil points that fall outside of domain
                        pyd = p + self.NZ*(self.NY-2)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,p)
                        self.coefs = np.append(self.coefs,-6.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pzu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pzd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyd)
                        self.coefs = np.append(self.coefs,1.0)

                    elif i!=0 and i!=(self.NX-1) and j==(self.NY-1) and k!=0 and k!=(self.NZ-1): #Nodes with PBC in the FACE y=L

                        #Redefine stencil points that fall outside of domain
                        pyu = p - self.NZ*(self.NY-2)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,p)
                        self.coefs = np.append(self.coefs,-6.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pzu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pzd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyd)
                        self.coefs = np.append(self.coefs,1.0)

                    elif i==0 and j==0 and k==(self.NZ-1): #Nodes with PBC in xy directions and Neuman in z direction in the CORNERS z=L

                        #Redefine stencil points that fall outside of domain
                        pxd = p + self.NZ*self.NY*(self.NX-2)
                        pyd = p + self.NZ*(self.NY-2)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,p)
                        self.coefs = np.append(self.coefs,-6.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pzd)
                        self.coefs = np.append(self.coefs,2.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.nbc = np.append(self.nbc,p)

                    elif i==(self.NX-1) and j==0 and k==(self.NZ-1): #Nodes with PBC in xy directions and Neuman in z direction in the CORNERS z=L

                        #Redefine stencil points that fall outside of domain
                        pxu = p - self.NZ*self.NY*(self.NX-2)
                        pyd = p + self.NZ*(self.NY-2)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,p)
                        self.coefs = np.append(self.coefs,-6.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pzd)
                        self.coefs = np.append(self.coefs,2.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.nbc = np.append(self.nbc,p)

                    elif i==0 and j==(self.NY-1) and k==(self.NZ-1): #Nodes with PBC in xy directions and Neuman in z direction in the CORNERS z=L

                        #Redefine stencil points that fall outside of domain
                        pxd = p + self.NZ*self.NY*(self.NX-2)
                        pyu = p - self.NZ*(self.NY-2)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,p)
                        self.coefs = np.append(self.coefs,-6.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pzd)
                        self.coefs = np.append(self.coefs,2.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.nbc = np.append(self.nbc,p)

                    elif i==(self.NX-1) and j==(self.NY-1) and k==(self.NZ-1): #Nodes with PBC in xy directions and Neuman in z direction in the CORNERS z=L

                        #Redefine stencil points that fall outside of domain
                        pxu = p - self.NZ*self.NY*(self.NX-2)
                        pyu = p - self.NZ*(self.NY-2)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,p)
                        self.coefs = np.append(self.coefs,-6.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pzd)
                        self.coefs = np.append(self.coefs,2.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.nbc = np.append(self.nbc,p)

                    elif i==0 and j!=0 and j!=(self.NY-1) and k==(self.NZ-1): #Nodes with PBC in x direction and Neuman in z direction in the EDGE z=L

                        #Redefine stencil points that fall outside of domain
                        pxd = p + self.NZ*self.NY*(self.NX-2)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,p)
                        self.coefs = np.append(self.coefs,-6.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pzd)
                        self.coefs = np.append(self.coefs,2.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.nbc = np.append(self.nbc,p)

                    elif i==(self.NX-1) and j!=0 and j!=(self.NY-1) and k==(self.NZ-1): #Nodes with PBC in x direction and Neuman in z direction in the EDGE z=L

                        #Redefine stencil points that fall outside of domain
                        pxu = p - self.NZ*self.NY*(self.NX-2)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,p)
                        self.coefs = np.append(self.coefs,-6.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pzd)
                        self.coefs = np.append(self.coefs,2.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.nbc = np.append(self.nbc,p)

                    elif i!=0 and i!=(self.NX-1) and j==0 and k==(self.NZ-1): #Nodes with PBC in y direction and Neuman in z direction in the EDGE z=L

                        #Redefine stencil points that fall outside of domain
                        pyd = p + self.NZ*(self.NY-2)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,p)
                        self.coefs = np.append(self.coefs,-6.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pzd)
                        self.coefs = np.append(self.coefs,2.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.nbc = np.append(self.nbc,p)

                    elif i!=0 and i!=(self.NX-1) and j==(self.NY-1) and k==(self.NZ-1): #Nodes with PBC in y direction and Neuman in z direction in the EDGE z=L

                        #Redefine stencil points that fall outside of domain
                        pyu = p - self.NZ*(self.NY-2)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,p)
                        self.coefs = np.append(self.coefs,-6.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pzd)
                        self.coefs = np.append(self.coefs,2.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.nbc = np.append(self.nbc,p)

                    elif i!=0 and i!=(self.NX-1) and j!=0 and j!=(self.NY-1) and k==(self.NZ-1): #Nodes with Neuman in z direction in the FACE z=L

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,p)
                        self.coefs = np.append(self.coefs,-6.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pzd)
                        self.coefs = np.append(self.coefs,2.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.nbc = np.append(self.nbc,p)

                    else: # All the rest of internal nodes.

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,p)
                        self.coefs = np.append(self.coefs,-6.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pzu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pzd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pxd)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyu)
                        self.coefs = np.append(self.coefs,1.0)

                        self.rows = np.append(self.rows,p)
                        self.columns = np.append(self.columns,pyd)
                        self.coefs = np.append(self.coefs,1.0)

                    p += 1


        #Coefficient matrix stored as Scipy sparse array
        self.coef_mat = csc_matrix((self.coefs,(self.rows,self.columns)),shape=(self.Ntot,self.Ntot))

        return self.coef_mat

    def update_elec_field(self):
        for particle in self.particles:
            X = particle.position[0] * self.N
            Y = particle.position[1] * self.N
            Z = particle.position[2] * self.N
            if Z <= self.compact_layer_limit * self.N and Z >= (self.simu_box_limit * self.N):
                elec_field_z_direction = self.elec_field[2][X][Y][Z-(self.simu_box_limit * self.N)]
                particle.elec_field = elec_field_z_direction
            elif Z > self.compact_layer_limit * self.N:
                elec_field_z_direction = self.elec_field[2][X][Y][(self.compact_layer_limit-self.simu_box_limit) * self.N]
                particle.elec_field = elec_field_z_direction

    def calc_electrode_surface_concentration(self):
        self.n_mv_nd_ef = 0
        self.n_mv_d_ef = 0
        for particle in self.particles:
            if particle.z >= self.Z - 1 and particle.particle_charge == 1:
                self.n_mv_nd_ef += 1
            elif particle.z >= self.Z - 1 and particle.particle_charge == 2:
                self.n_mv_d_ef += 1

        self.conc_mv_nd_ef = self.n_mv_nd_ef / (N_A * self.electrode_surface)
        self.conc_mv_d_ef = self.n_mv_d_ef / (N_A * self.electrode_surface)
        return self.conc_mv_nd_ef, self.conc_mv_d_ef

    def calc_potential(self):
        self.calc_conc_MV()
        self.phi_inter = np.mean(self.phi, axis=(0, 1))[self.NZ-1]
        self.elec_E = np.mean(self.elec_field)
        self.electrode_potential = self.potential_drop + self.phi_inter

    def clean_outfiles(self):
        with open(self.file_xyz, 'w') as f:
            f.write('')
        with open(self.file_pot, 'w') as f:
            f.write('      Time      c_MV+     c_MV2+     c_MV0       E_interface     Potential(V)          Delta_E       Capacitance\n')
        with open(self.file_psd, 'w') as f:
            f.write('      Time          Del_t          J_Far      ThetaFree          Sigma         Gamma      '
                    'Overpotential  Phi_inter            Event \n')

    def write(self):
        xyz = (f'{len(self) + 8}\n'
               'Type  x  y  z\n')

        box_top = ((0, 0, self.Z), (0, self.Y + 1, self.Z),
                   (self.X + 1, 0, self.Z), (self.X + 1, self.Y + 1, self.Z),
                   (0, 0, 0), (self.X+1, 0, 0), (0, self.Y+1, 0), (self.X+1, self.Y+1, 0))

        for x, y, z in box_top:
            xyz += f'{5:4d} {x:2d} {y:2d} {z:2d}\n'
        #   pxyz += f'{0:4d} {x:2d} {y:2d} {z:2d}\n'

        for particle in self:
            xyz += f'{particle}\n'

        pot = f'{self.time:10.4e} {self.conc_mv_charged:10.4f} {self.conc_mv_discharged:10.4f} ' \
              f'{self.conc_mv_degradate:10.4f} {self.reaction_plane_E:16.4e} {self.electrode_potential:16.4e}\n'

        psd = f'{self.time:10.4e} {self.dt:14.4e} {self.j_far:14.4e} {self.theta_free:14.4e} {self.charge_density:14.4e}' \
              f'{self.big_gamma:14.4e} {self.potential_drop:14.4e} {self.phi_inter:14.4e} ' \
              f'{self.event_type:>16} {self.discharge_rate}\n'

        with open(self.file_xyz, 'a') as f:
            f.write(xyz)

        with open(self.file_pot, 'a') as f:
            f.write(pot)

        with open(self.file_psd, 'a') as f:
            f.write(psd)

    def __len__(self):
        return len(self.grid)

    def __iter__(self):
        for sub_particle in self.grid.values():
            yield sub_particle

    def __str__(self):
        return
