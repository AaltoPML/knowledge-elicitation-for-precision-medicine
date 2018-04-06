# Improving genomics-based predictions for precision medicine through active elicitation of expert knowledge

This repository contains the implementation of the computational methods in the paper Sundin et al., Improving genomics-based predictions for precision medicine through active elicitation of expert knowledge [1].

Main files:

 * `code/linreg_sns_ep.m` contains the expectation propagation inference algorithm for the sparse linear regression model given training data and user feedback.
 * `code/compute_utilities.m` contains the experimental design computations to estimate the utilities for choosing the next query for the user.
 * `simulation_example/run.m` contains an example of an experiment with simulated data and simulated user feedback.

## Reference

If you use this code, please cite the paper:

[1] Iiris Sundin\*, Tomi Peltola\*, Luana Micallef, Homayun Afrabandpey, Marta Soare, Muntasir Mamun Majumder, Pedram Daee, Chen He, Baris Serim, Aki Havulinna, Caroline Heckman, Giulio Jacucci, Pekka Marttinen, and Samuel Kaski. **Improving genomics-based predictions for precision medicine through active elicitation of expert knowledge.** Accepted to the ISMB 2018 conference proceedings, to be published in Bioinformatics.

\* Equal contribution.

## Contact

 * Iiris Sundin, iiris.sundin@aalto.fi
 * Tomi Peltola, tomi.peltola@aalto.fi
 * Pekka Marttinen, pekka.marttinen@aalto.fi
 * Samuel Kaski, samuel.kaski@aalto.fi

## Acknowledgements

`hermitepolynomial.m` is from [EKF/UKF toolbox](http://becs.aalto.fi/en/research/bayes/ekfukf/) (GPLv2 or later) and written by Arno Solin.

`dirrand.m` is from [GPStuff toolbox](http://research.cs.aalto.fi/pml/software/gpstuff/) (GPLv3 or later) and written by Aki Vehtari.

This work was supported by the Academy of Finland [Finnish Center of Excellence in Computational Inference Research COIN, grant numbers 295503, 294238, 292334, 284642, 305780, 286607, 294015]; by Jenny and Antti Wihuri Foundation; and by Alfred Kordelin Foundation.

## License

GPLv3
