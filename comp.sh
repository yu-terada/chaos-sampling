#! /bin/bash

for g in 8.0 # gain parameter in the initial internal weights
do
    for D in 1.0 # noise strength in perturbation learning
	     do
		 for flag in 1 3 # 1 changes the internal and readout weights while 3 changes only the readout weights
		 do
		     for seed in 6 # seed of random number generator
		     do
			   ./comp_sub.sh ${g} ${D} ${flag} ${seed}
			 done
	    done
    done
done
