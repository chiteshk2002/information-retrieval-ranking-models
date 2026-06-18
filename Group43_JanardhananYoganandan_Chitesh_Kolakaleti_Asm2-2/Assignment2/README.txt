IFN647 ASSIGNMENT 2 – USER MANUAL

Group:
43

Student Names:
Janardhanan Yoganandan – N11918411
Chitesh Kolakaleti – N11907304


1. PROJECT OVERVIEW

This project implements and evaluates three information retrieval ranking models for IFN647 Assignment 2.

The three models are:

1. Baseline1 – BM25-based information retrieval model
2. Baseline2 – Jelinek-Mercer smoothing language model
3. Model_C – Custom pseudo-relevance based hybrid ranking model

The models rank documents for topics R101 to R150 using the corresponding datasets Dataset101 to Dataset150. The ranking outputs are saved as .dat files in the ModelOutputs folder.

The project also evaluates the three models using:
- Average Precision and MAP
- Precision@10
- DCG@10

A paired significance test is also conducted to compare Model_C with Baseline1 and Baseline2.


2. REQUIRED SOFTWARE

Python version:
Python 3.x

External packages:
No external package installation is required.

The implementation uses Python built-in libraries only:
- os
- re
- glob
- zipfile
- string
- collections
- csv
- math

The project also includes:
- common-english-words.txt
- stemming/porter2.py

The stop-word file and Porter2 stemming file are included in the submitted folder and are used during document preprocessing.


3. PROJECT FOLDER STRUCTURE

The submitted folder should contain:

Task1_Baseline1_BM25.py
Task1_Baseline2_JM.py
Task2_ModelC.py
Task3_Run_All_Models.py
Task3_Data_Loader.py
Task3_Output_Generator.py
Task4_Evaluation.py
Task5_Significance_Test.py
Task6_ApplicationScenario.txt
README.txt
common-english-words.txt
stemming/
data/
ModelOutputs/
EvaluationOutputs/


4. FILE DESCRIPTIONS

Task1_Baseline1_BM25.py
Implements Baseline1 using the BM25 ranking model. It uses k1 = 1.2, k2 = 500 and b = 0.75.

Task1_Baseline2_JM.py
Implements Baseline2 using the Jelinek-Mercer smoothing language model. It uses lambda = 0.3.

Task2_ModelC.py
Implements Model_C. The model uses pseudo-relevance feedback, query expansion and a hybrid score based on BM25 and JM ranking evidence.

Task3_Run_All_Models.py
Main execution file. It runs Baseline1, Baseline2 and Model_C for all topics R101 to R150, writes ranking outputs, runs evaluation and produces significance test results.

Task3_Data_Loader.py
Loads topics, stop words, document collections and relevance judgement files. It also parses XML documents and represents documents using document ID, term-frequency dictionary and document length.

Task3_Output_Generator.py
Writes ranking output files and Model_C expansion term output.

Task4_Evaluation.py
Calculates Average Precision, MAP, Precision@10 and DCG@10.

Task5_Significance_Test.py
Calculates paired t-test results comparing Model_C with Baseline1 and Baseline2.

Task6_ApplicationScenario.txt
Notes that Task 6 is answered in the final report. No Python execution is required for Task 6.

common-english-words.txt
Stop-word list used during preprocessing.

stemming/porter2.py
Porter2 stemming file used during preprocessing.


5. HOW TO RUN THE PROJECT

Step 1:
Open the project folder in PyCharm or another Python IDE.

Step 2:
Make sure the data folder is named "data" and is placed inside the project folder next to the Python files. The data folder must contain the topics file and the document and relevance-judgement data, in either of the following two forms.

Extracted form (already provided in this submission):

data/Topics.txt
data/Doc_Collection/Dataset101 ... data/Doc_Collection/Dataset150
data/Relevant_Judgements/Dataset101.txt ... data/Relevant_Judgements/Dataset150.txt

Zipped form:

data/Topics.txt
data/Doc_Collection.zip
data/Relevant_Judgements.zip

If only the zipped form is present, the program automatically extracts Doc_Collection.zip and Relevant_Judgements.zip into the data folder on the first run. If the extracted Doc_Collection and Relevant_Judgements folders already exist, the extraction step is skipped.

Step 3:
Run the main file:

python Task3_Run_All_Models.py

or:

python3 Task3_Run_All_Models.py


6. EXPECTED OUTPUTS

After running Task3_Run_All_Models.py, the following outputs are generated.

A. ModelOutputs folder

For each topic R101 to R150, the program creates:

Baseline1_R101_Ranking.dat to Baseline1_R150_Ranking.dat
Baseline2_R101_Ranking.dat to Baseline2_R150_Ranking.dat
ModelC_R101_Ranking.dat to ModelC_R150_Ranking.dat

Each ranking file uses the required format:

Doc_ID Score

Example:

46974 1.8878533347166448
46547 1.8878533347166448
62325 1.4153502848719022


B. EvaluationOutputs folder

per_topic_evaluation.csv
Contains per-topic Average Precision, Precision@10 and DCG@10 for Baseline1, Baseline2 and Model_C.

evaluation_summary.csv
Contains the average evaluation results for each model.

significance_tests.csv
Contains t-statistic and p-value results for Model_C vs Baseline1 and Model_C vs Baseline2.

top10_rankings.csv
Contains the top 10 ranked documents for each topic and each model.

model_c_expansion_terms.txt
Contains the expansion terms selected by Model_C.


7. CURRENT EVALUATION SUMMARY

The current evaluation summary is:

Model        MAP        Average_P@10   Average_DCG@10
Baseline1    0.491078   0.358000       2.090029
Baseline2    0.458335   0.344000       1.942974
Model_C      0.497682   0.372000       2.149772

These results are saved in:

EvaluationOutputs/evaluation_summary.csv


8. TASK COVERAGE

Task 1:
Implemented in Task1_Baseline1_BM25.py and Task1_Baseline2_JM.py.

Task 2:
Implemented in Task2_ModelC.py.

Task 3:
Implemented and executed through Task3_Run_All_Models.py.

Task 4:
Implemented in Task4_Evaluation.py, with results saved in EvaluationOutputs.

Task 5:
Implemented in Task5_Significance_Test.py, with results saved in EvaluationOutputs/significance_tests.csv.

Task 6:
Answered in the final report. No Python output is required for this task.
