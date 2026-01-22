# Ising

The project is to reproduce some results of the papers :
- MACHINE LEARNING OF NONEQUILIBRIUM PHASE TRANSITION IN AN ISING MODEL ON SQUARE LATTICE
- Detection of phase transition via convolutional neural networks
and maybe in a second time to generates samples with ML methods.

### Generate Data
We focussed on ising model simulations based on metropolis algorithm. With many small tricks we finally are able to generate many samply with fast execution time.
From this, we are able to reproduce few already explored properties of the 2D ising model.


<img height="300" src="https://github.com/joseph-elf/Ising/blob/main/IMGs/T%3D0.10%2CH%3D0.00.png">
<img height="300" src="https://github.com/joseph-elf/Ising/blob/main/IMGs/T%3D1.00%2CH%3D0.00.png">

<img height="300" src="https://github.com/joseph-elf/Ising/blob/main/IMGs/magnetization.png">
<img height="300" src="https://github.com/joseph-elf/Ising/blob/main/IMGs/oneGridCorr.png">
<img height="300" src="https://github.com/joseph-elf/Ising/blob/main/IMGs/severalCorr.png">


### Find T_c
We are currently implementing a CNN used to classify samples (ordered, unordered) and to predict T_c

### Generate samples ?
