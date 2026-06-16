# English to Navajo Machine Translation

## Prompting Experiments

To conduct prompting experiments, select which directory you intend to use (UMR or morphosyntactic) and then run the files in numerical order (begining ith the file that starts with "1_") and proceeding. Each python script will produce a file or a .out file which should be verified at every step to confirm that results are proceeding as desired. 

## Fine-tuning Experiments 

To conduct fine-tuning experiments, select the directory which corresponds to the model and type of fine-tuning you are looking to complete (i.e. if you want to do NSL fine-tuning on NLLB, select NSL_nllb_ft). Then, change the following variables in the .sh script to train your desired model:
EXPERIMENT_NAME=
TARGET_LANG=
SYLLABIFY=
MODEL_SIZE=
Run the .sh file and your model will be saved to a directory with the experiment name you selected.

### TODO: 
There are two major edits that need to be completed before rerunning the fine-tuning experiments to ensure we have no data leakage:
1) In each of the fine-tuning directories there is a file called "my_english.txt" and "my_navajo.txt", these files need to be cleared and only contain the morphosyntactic data. The morphosyntactic data is a superset of the UMR dataset and currently "my_english.txt" and "my_navajo.txt" are compilations of BOTH the UMR and morphosyntactic data. This leads to 506 overlapping sentences which could then be split across both training and test sets and therefore creates data leakage. To avoid this just remove the UMR data from these files. THIS NEED TO BE DONE IN ALL FOUR FINE-TUNING DIRECTORIES!!! If I am not mistake, you should just be able to remove the first 506 sentences in both files, but that needs to be verified.
2) Add a method right before the data is split in the train_"model".py file that verifies there are no duplicate sentences. This needs to loop through ALL the data and ensure each sentence is unique. Note, there are duplicate sentences in the Bible (specifically 26 if I am not mistaken). This is the major edit that needs to be done and it needs to be done across all fine-tuning directories. It is a small edit in terms of code, but a massive edit in terms of results.
Once the above edits are completed you will run the following experiments:
  1) NLLB gle_Latn Syllabified
  2) NLLB gle_Latn No Syllabification
  3) NLLB grn_Latn Syllabified
  4) NLLB grn_Latn No Syllabification
  5) NLLB fin_Latn Syllabified
  6) NLLB fin_Latn No Syllabification
  7) M2M100 mg Syllabified
  8) M2M100 mg No Syllabification
  9) M2M100 sw Syllabified
  10) M2M100 sw No Syllabification
  11) M2M100 fi Syllabified
  12) M2M100 fi No Syllabification
You will do these by changing the .sh scripts, specifically these variables in them:
EXPERIMENT_NAME=
TARGET_LANG=
SYLLABIFY=
Check the .out and .err files to ensure everything appears as it is suppose to, specifically check the five fold split looks correct, the syllabification is working, and that you are running the correct model and specifications. There is a lot of debugging in the files to ensure everything is working properly, so it is important to use it!
Then, once these are completed and tested (using the test files in each of the directories), you will select the best configuration (likely NLLB grn_Latn No Syllabification and M2M100 sw No Syllabification) for each model and then run that on the NSL_ft directories. THE EDITS ABOVE MUST ALSO BE APPLIED TO THESE DIRECTORIES!!! Make sure edits #1 and #2 are applied here as well.
Email me with any questions: emarkle26@amherst.edu
