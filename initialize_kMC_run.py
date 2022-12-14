from particles_CC import Board
import logging
from configparser import ConfigParser


def read_ini_file(filename):

    """
    This function reads parameters entered by the user in the *.ini file,
    converts the important figures mentioned as strings into float in the *.ini file
    then stores the information as dictionaries for the main part of the simulation to use.
    These dictionaries, namely, simulation_info and run_info are returned
    :param filename:
    :return:
    """
    config = ConfigParser()
    config.read(filename)

    board_info = dict(config["BOARD INFO"])
    board_info['board size'] = tuple(map(int, board_info['board size'].split()))
    board_info['unit scale'] = float(board_info['unit scale'])

    operation_conditions = dict(config["OPERATION CONDITIONS"])
    operation_conditions['temperature'] = float(operation_conditions['temperature'])
    operation_conditions['input current density'] = float(operation_conditions['input current density'])

    electrolyte_info = dict(config["ELECTROLYTE INFO"])
    electrolyte_info['mv initial concentration'] = float(electrolyte_info['mv initial concentration'])
    electrolyte_info['mv initial viscosity'] = float(electrolyte_info['mv initial viscosity'])
    electrolyte_info['mv solution permittivity'] = float(electrolyte_info['mv solution permittivity'])
    electrolyte_info['mv initial soc'] = int(electrolyte_info['mv initial soc'])
    electrolyte_info['solution initial concentration'] = float(electrolyte_info['solution initial concentration'])
    electrolyte_info['cl radius'] = float(electrolyte_info['cl radius'])
    electrolyte_info['cl charge'] = int(electrolyte_info['cl charge'])

    electrochemistry_info = dict(config["ELECTROCHEMISTRY INFO"])
    electrochemistry_info['oxidation reorganization energy'] = float(electrochemistry_info['oxidation reorganization energy'])
    electrochemistry_info['reduction reorganization energy'] = float(electrochemistry_info['reduction reorganization energy'])
    electrochemistry_info['degradation energy'] = float(electrochemistry_info['degradation energy'])
    electrochemistry_info['compact layer thickness'] = float(electrochemistry_info['compact layer thickness'])
    electrochemistry_info['compact layer permittivity'] = float(electrochemistry_info['compact layer permittivity'])
    electrochemistry_info['tunneling distance'] = float(electrochemistry_info['tunneling distance'])

    MV_info = dict(config["MV INFO"])
    MV_info['mv molar mass'] = float(MV_info['mv molar mass'])
    MV_info['mv gyration radius before discharging'] = float(MV_info['mv gyration radius before discharging'])
    MV_info['mv gyration radius after discharging'] = float(MV_info['mv gyration radius after discharging'])
    MV_info['mv standard electrode potential'] = float(MV_info['mv standard electrode potential'])
    MV_info['mv charged valence'] = int(MV_info['mv charged valence'])
    MV_info['mv discharged valence'] = int(MV_info['mv discharged valence'])

    run_info = {'max_iterations': config['RUN INFO'].getint('iterations'),
                'frequency': config['RUN INFO'].getint('frequency'),
                'time': config['RUN INFO'].getfloat('time')}

    simulation_info = dict(locals())

    return run_info, simulation_info

def initialize(args):
    """
    Passed the user define information to the class Board.
    Other classes particle and subparticle are initiated from inside Board.
    board is an instance of class Board
    :return:
    """
    print('initialize')
    run_info, simulation_info = read_ini_file(args.configfile)
    if args.jobname:
        run_info['name'] = args.jobname
    logging.basicConfig(filename=run_info['name'] + '.log', filemode='w', level=logging.INFO)
    board = Board(**simulation_info)
    board.clean_outfiles()
    return run_info, board

def run(board):
    print('run')
    """
    :param board: all the functions related to the kMC model is coed in the class board
    :param max_iterations: write the output file every max_iterations
    :param frequency: update the Poisson electric field calculation every frequency
    :param name: output name
    :return:
    """
    iteration = 1

    while board.time < board.cut_off_time:
        print(f'\rProgress: {board.total_soc}', end='')
        board.j_input = board.j_input_dis
        board.vssm_sonar()

        if iteration % board.frequency == 0 or board.event_type == 'MV_Discharge' or board.event_type == 'MV_Charge':
            board.define_b_matrix()
            board.solve_poisson()
            board.update_elec_field()

        if iteration % board.iteration == 0 or board.event_type == 'MV_Discharge' or board.event_type == 'MV_Charge':
            board.write()
        iteration += 1
    return