####################################################################
      ViWi Vision-Aided Beam Tracking (ViWi-VABT) Challenge
####################################################################

# Result file format:
#####################

Test results of the challenge are expected to be submitted in a csv file. The file must have the following structure:
Number of lines should be equal to the number of data samples, and for each sample (line), there should be 9 numbers. 
The 9 numbers should provide, respectively, the predicted beams for the three-tier task. First number is the index of 
the first future beam. The second to fourth numbers should be for the predicted 3 future beams. Finally the fifth to 
ninth numbers should be for the predicted 5 future beams. 

IMPORTANT NOTES:
1) The order of the samples (lines) in your submistted result file MUST match that of the 
samples in the test set. The following is meant as an example:

In the csv test set file (testset_evaluation.csv), the data sample have the following structure

    | Beam1 Beam2 ... Beam8   img path1   ...   Img path 8  |
    | ----------------------------------------------------- |
    |   1     4   ...   2    ./image1.jpg ...  ./image5.jpg | --> sample 1
    | ----------------------------------------------------- |
    .                                                       .
    .                                                       .
    | ----------------------------------------------------- |
    |  2      5   ...   5   ./image1.jpg ...  ./image5.jpg  |--> sample X
    |_______________________________________________________|

The results file must have the following structure:

    | val.1 val.2 val.3 val.4 val.5 val.6 val.7 val.8 val.9 --> sample 1 
    .
    .
    | val.1 val.2 val.3 val.4 val.5 val.6 val.7 val.8 val.9 --> sample X
    

2) The values (val.1, ..., val.9) in the result file must all be integers, i.e., no decimal points (e.g., 2.0) or fractions (e.g., 0.5). 

3) The submistted file should be named as follows: "team_id.csv"  --> just replace team_id with you team's id number


###########################################################################################################################################
Should you have more questions, please do not hesitate to contact us at: viwi.mlc@gmail.com

From the ViWi-BT challenge team,
Thank you and good luck