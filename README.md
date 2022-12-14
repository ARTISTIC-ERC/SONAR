# SONAR kinetic Monte Carlo model

PROJECT SUMMARY
This python coded model aims to simulate the discharge of the anode materials in a redox flow battery system. The target material is methyl viologen. 
The discharging process is simulated by kinetic Monte Carlo algorithm. In each iteration, the algorithm chooses one event to execute and update the system configuration.
The main function is written in the 'simulate_SONAR_RFB.py'. The related kinetic Monte Carlo algorithm is coded in the 'particles_CC.py'. 
The 'initialize_kMC_run.py' is for initialization. The input parameters can be altered in 'user_input_SONAR.ini'

OUTPUT FILES
There will be ofur output files, once the code is launched. 
The output file with '.txt' recorded the simulation results, including the simualtion time, time step, electrode charge density, 
the potential drop thorugh the compact layer, the potential on the interface between the compact layer and the diffuse layer, and the type of events.
The output file with '.xyz' recorded the location of each ions. To visualise the 3D configuration of the simualtion box, users can use the software Ovito to process.
the output file with '.out

POST PROCESSING
related post-processing code will be added later

HOW TO LAUNCH
launch in terminal with the command line
'python simulate_SONAR_RFB.py'

ACKNOWLEDGEMENT
This model is developed under the funding of the European project SONAR, under the grant agreement 875489

TO CITE:
Yu, J., Shukla, G., Fornari, R. P., Arcelus, O., Shodiev, A., de, P., Franco, A. A., 
Gaining Insight into the Electrochemical Interface Dynamics in an Organic Redox Flow Battery with a Kinetic Monte Carlo Approach. 
Small 2022, 18, 2107720. https://doi.org/10.1002/smll.202107720
