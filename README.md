# summarization-models
Project for the course NLU &amp; NLG, MSc Language Technology.

This repo consists the code for developing three summarization systems for the CNN v3 dataset:  https://huggingface.co/datasets/cnn_dailymail/viewer/3.0.0

**System_1**: Lead 2 (baseline)  A summarizer that just extracts and returns the first two sentences of each input text.
**System_2**: Feature based extractive summarizer trained with Supervised Machine Learning (Logistic Regression with GridSearch Hyperparameter tuning)

**System_3**: bert_extractive_summarizer (No training, direct inference on the test set)
**System_4**: Finetuning T5-base for Text summarization
