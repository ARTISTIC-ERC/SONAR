# SONAR kinetic Monte Carlo model
![sonar](https://user-images.githubusercontent.com/60663976/208126144-5d568843-618f-4a72-9dd7-5c76fc88bc09.png)

## SONAR PROJECT
In search of truly competitive solutions for storing energy from renewable resources, the SONAR-team sets out to develop a framework for the simulation-based screening of electroactive materials for organic redox flow batteries (RFBs) – in aqueous and non-aqueous solutions. This will help to reduce the cost and time-to-market and thus strengthen the competitiveness of the EU’s battery industry.

We follow a multiscale modelling paradigm, starting from the automated generation of candidate structures for the electroactive material and then iterate through molecular-, electrochemical interface-, porous electrodes-, cell-, stack-, system- and techno-economic-level models.
Finally, storage technologies are only comparable when using the levelized-cost-of-storage (LCOS) as a global metric, which accounts for all relevant effects across all the scales.

The simulation results go into a database for further processing – we will exploit advanced data integration, analysis and machine-learning techniques, drawing on the growing amount of data produced during the project in order to speed up the computations.
Selected models will be validated by measurements in RFB half-cells and lab-sized test cells to ensure our predictions' quality. We will work closely with industrial partners to ensure the commercial viability of the results.

## PROJECT SUMMARY
This python coded model aims to simulate the discharge of the anode materials in a redox flow battery system. The defaut material is methyl viologen. 
The discharging process is simulated by kinetic Monte Carlo algorithm. In each iteration, the algorithm chooses one event to execute and update the system configuration.
The main function is written in the 'simulate_SONAR_RFB.py'. The related kinetic Monte Carlo algorithm is coded in the 'particles_CC.py'. 
The 'initialize_kMC_run.py' is for initialization. The input parameters can be altered in 'user_input_SONAR.ini'

## KMC MODEL
The kinetic Monte Carlo model adopted in this work is called the Variable Step Size Method. The algorithm first identified all the possible events based on the current configuration of the simulation box and then randomly chooses an event to execute in each iteration. The configuration and the characteristics of the system is then updated according to the chosen event. The cutoff condition is the simulation time.

## DOUBLE LAYER MODEL
The Double Layer model is immplemented to capture the formaiton of the electrical double layer, and its impact on ion motions. The electrical double layer is considered as seperated zones of compact layer and diffuse layer. The electrode potential is the sum of the potential drop thorugh the compact layer and the potential on the interface between the diffuse layer and the compact layer. The compact layer is considered as a capacitor. and electrical field distribution in the diffuse layer is solved by Poisson Equation.   

## OUTPUT FILES
There will be three output files. 
The output file with '.txt' records the simulation results, including the simualtion time, time step, electrode charge density, the potential drop thorugh the compact layer, the potential on the interface between the compact layer and the diffuse layer, and the type of events.
The output file with '.xyz' records the location of each ions. To visualise the 3D configuration of the simualtion box, users can use the software Ovito to process.
The output file with '.log' is the logfile which doesn't contain simulation results.

## POST PROCESSING
Related post-processing code will be released later

## HOW TO LAUNCH
Launch in terminal with the command line:
```
python simulate_SONAR_RFB.py'
```

#### AUTHORS
- **Jia Yu**
- **Garima Shukla**
- **Oier Arcelus**
- **Alejandro A. Franco**

#### ACKNOWLEDGEMENT
This model is developed under the funding of the European project SONAR, under the grant agreement 875489

#### TO CITE:
Yu, J., Shukla, G., Fornari, R. P., Arcelus, O., Shodiev, A., de, P., Franco, A. A., 
Gaining Insight into the Electrochemical Interface Dynamics in an Organic Redox Flow Battery with a Kinetic Monte Carlo Approach. 
Small 2022, 18, 2107720. https://doi.org/10.1002/smll.202107720

#### COPYRIGHT
© Université de Picardie Jules Verne & CNRS (Laboratoire de Réactivité et Chimie de Solides) -  <alejandro.franco@u-picardie.fr>
This model is for academic use only, for commercialization request, please contact Prof. Alejandro A. Franco.