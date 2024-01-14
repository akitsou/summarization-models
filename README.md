# summarization-models
Project for the course NLU &amp; NLG, MSc Language Technology.

This repo consists the code for developing three summarization systems for the CNN v3 dataset:  https://huggingface.co/datasets/cnn_dailymail/viewer/3.0.0

**System_1**: Lead 2 (baseline)  A summarizer that just extracts and returns the first two sentences of each input text.

**System_2** (Winner model): Feature based extractive summarizer trained with Supervised Machine Learning (Logistic Regression with GridSearch Hyperparameter tuning)
Scripts for portion generation, Preprocessing, Feature extraction, ML models, Summary Generation, Downsampling experiment can be found in **Task2_Three-steps** directory

**System_3**: [bert_extractive_summarizer](https://pypi.org/project/bert-extractive-summarizer/) (No training, direct inference on the whole test set)

**System_4**: Finetuning T5-base for Text summarization (3500 train, 1500 validation, 5500 test set)
- MAX_LEN= 512
- SUMMARY_LEN=150
- MAX_EPOCHS=5
- batch_size=8

# Contributor Expectations
Machine Learning
1. Experiment with other Machine Learning Algorithms: SVM, Decision Trees, Random Forest and especially with enselmble models, like AdaBoost Classifier.
2. Add more features on training: normalized Named Entity Count (sentence NE count/sentence word count) and Verb Count (sentence Verb count/sentence word count)
3. Use feature selection techniques to find the n most informative features

LLMs
1. Increase train and validation sizes, and hyperparameter values: epochs, batch size
2. Use larger models like: T5_large, BART_large, GPT models
3. Prompt Chat GPT
4. Try extractive summarization with DistilBERT

