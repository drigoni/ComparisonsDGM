# Reference
This folder refers to the code used in the document: [Confined generation of semantically valid
Graphs through the regularization of variable autoencoders](https://arxiv.org/pdf/1809.02630.pdf).
The code has been kindly sent to us by e-mail by the authors, but for reasons related to the ownership of the code, we are not allowed to publish it.

# Changes and New Files
Below are reported the changes made to the original code to obtain the version used in the paper for comparisons.
* All the functions for reading datasets and generating molecules have been added;
* since the model sent to us had been configured for learning on the QM9 data set,  we proceeded to modify it for learning on ZINC;
* in the received implementation it was possible to train the model with different loss functions presented in the authors' article. 
After performing the training with the `cap` (only valence) and `conc` (valences and losses on connections) loss function and running the tests on the generated molecules, 
we decided to present only the version that uses the `cap` constraints as has presented better results;
* Train on `ZINC`: 300 epochs are used for the `cap` version and 400 for the version without constrains;
* Train on `QM9`: we used the number of epochs indicated by the authors: 150 for the `cap` version and 200 for the version without constrains.


# Errors in the Original Code
During the implementation of the test functions, an error was discovered in the original code that drastically reduced the value of the reconstruction metric.
This error was due to an incorrect comparison between molecular structures, which sometimes identifies two identical molecules even though they had a different number of atoms.
