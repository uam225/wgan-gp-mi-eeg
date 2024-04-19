# bsc-project-source-code-files-2023-24-uam225
Condtiional WGAN for EEG Data Synthesis

This is the repository for the Wasserstein Generative Adversarial Network (WGAN) designed to generate synthetic motor imagery EEG data. Developed as part of the final year project for BSc Data Science & Computing at Birkbeck College, University of London.

This README provides instructions on how to use the various scripts included in the repository to process EEG data, train the GAN, and evaluate its performance.



Data Preparation
Download Data: First, download the necessary EEG data from (https://figshare.com/articles/code/shu_dataset/19228725/1).

Feature Extraction: Run the feature extractor available on this repository, which is adapted from the dataset's toolbox, to extract relevant channels and save the CSP features.

Training the GAN
Train the Model: In train_GAN.py, choose the model configuration you wish to run by uncommenting the required lines. The generated samples will be saved in the specified directory.

Classification and Label Assignment
Run SVM Classification: Use SVM.py to train a classifier on the original data and to predict labels for the generated data.

Label Assignment: After classification, run label_assignment.py to assign the learned labels to the generated data.

Visualisation
Original Data: Use visualiser_original_data.py to view the original data.
Generated Data: Use visualiser.py to plot the generated data for comparison with the original data.

Evaluation
Evaluate the Model: Use wgan_eval_and_save.py to evaluate the trained model. The evaluation results and any additional outputs will be saved accordingly.

Additional Resources
Plotting and Evaluating: The vis_and_eval folder contains additional scripts for plotting and evaluating the model.