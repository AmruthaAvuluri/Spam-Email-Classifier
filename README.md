# Spam Email Classifier using Machine Learning

This project implements a Spam Email Classifier using Python and machine learning techniques. The main objective of the project is to automatically classify messages as spam or ham (not spam) based on their textual content. It uses a supervised learning approach with a labeled dataset.

The text data is transformed into numerical form using TF-IDF (Term Frequency–Inverse Document Frequency) vectorization. A Multinomial Naive Bayes classifier is then trained on the processed data to learn patterns commonly found in spam messages. The trained model is evaluated using a test dataset, and the accuracy score is displayed.

The project also demonstrates how the trained model can be used to predict the category of a new message provided by the user. This project is suitable for beginners and helps in understanding text preprocessing, feature extraction, model training, and evaluation in machine learning.

The project is implemented using Python with libraries such as Pandas and Scikit-learn. To run the program, ensure Python and the required libraries are installed, place the dataset file in the same directory as the script, and execute `python task1_spam.py`.

Author: Amrutha Reddy
