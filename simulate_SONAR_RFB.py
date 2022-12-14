from argparse import ArgumentParser

from initialize_kMC_run import initialize, run


def getargs(argv=''):
    parser = ArgumentParser()
    parser.add_argument('jobname', nargs='?')
    parser.add_argument('-c', '--config', dest='configfile', default='user_input_SONAR.ini')
    args = parser.parse_args(argv.split()) if argv else parser.parse_args()
    return args


def main(argv=''):
    args = getargs(argv)
    run_info, board = initialize(args)
    board.write()
    try:
        run(board)
    except Exception as e:
        board.write()
        raise e

'''Change the first value to modify the output file name'''
if __name__ == '__main__':
    main('simulation_name -c user_input_SONAR.ini')
