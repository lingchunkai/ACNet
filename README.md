This folder contains a minimal working example (MWE) for ACNet. 

We include the core code for ACNet, as well as code for one (sub)-experiment
for each of the 3 subsections in Section 5. 

Plotting of graphs was done by separately sampling and evaluating pdfs in ACNet,
followed by calling pgfgen. These components are not include in this MWE.

Files
=====
The main files are `main.py` and `dirac_phi.py`. 

`main.py`
This contains the code performing the generation of the computational
graph, Newton's root finding method, bisection method (for root finding in the 
case of conditional sampling), as well as the implementation for the 
gradients of inverses. 

`dirac_phi.py`
This contains the code defining the specific structure of phi_NN. 

Dependencies:
    pytorch
    numpy
    matplotlib
    scipy
    optional: sacred, used for monitoring and keeping track of experiments
    optional: scikit-learn, used only to download the boston housing dataset.

Other files are:
`phi_listing.py`
Contains definitions for other phi's for the commonly used copula. Used to generate
synthetic data.

The rest of the files contain boilerplate code and helper functions.

Instructions
============
We asume that dependencies are installed correctly. 

Experiment 1: Synthetic Data
----------------------------
To generate synthetic data drawn from the Clayton distribution, first run:

>> python -m gen_scripts.clayton -F gen_clayton

This will generate synthetic data in the pickle file `claytonX.p`. 
If this is your first run, X=1.

Next, navigate to train_scripts/clayton.py. In the cfg() function, 
modify the data_name accordingly. If this is your first run, the
default should be alright. Then, run:

>> python -m train_scripts.clayton -F learn_clayton

The experiment should run, albeit with slightly different values as us due to slight differents 
in randomness. The network should converge after around 10k (training) iterations.

After every fixed interval, samples will be drawn from the learned network and plotted
in the `/sample_figs` folder. 

Experiment 2: Real World Data
-----------------------------
The boston housing dataset will automatically be downloaded using scikit learn. Simply run 

>> python -m boston.train -F boston_housing

to run the experiment. Sampled points will be found in `/sample_figs` in the appropriate
folder. 

Note that since the dataset is fairly small, results may vary significantly between runs. You may change
the train/test split in `/boston/train.py` directly to get varying results. In our experiments
we used 5 different seeds and took averages.

Typically, convergence occurs at around 10k epochs. Note that because the dataset is so small, in
some settings, test loss will be *better* than training loss. This is *not* a bug.

Experiment 3: Noisy Data: Synthetic Data
----------------------------------------
Generate the data from the Clayton copula as shown in the first section. Simply run

>> python -m train_scripts.clayton_noisy -F learn_clayton_noisy

As before, samples from the learned distribution will be periodically saved in `/sample_figs`.

Note that the training loss being reported are the log *probabilities* and not the log *likelihoods*. 
However, the test loss is based on log *likelihoods*, in order to facilitate comparison with the non-noisy
case (Experiment 1). 

In order to change the magnitude of noise, modify the variable `width_noise` in `train_scripts/clayton_noisy.py`.
This coressponds to \lambda in the paper.

