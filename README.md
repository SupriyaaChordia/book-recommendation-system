# Book Rating Prediction and Genre Classification

Exploring book preferences through interaction data and review text.

[View implementation](https://github.com/SupriyaaChordia/book-recommendation-system/blob/main/Book_reccomender.py) · [Analytics portfolio](https://supriyaachordia.github.io/portfolio/)

## Overview

This machine-learning project addresses two tasks relevant to book discovery: predicting a user's rating for a book and classifying a review into one of five genres. The implementation combines collaborative filtering with text classification.

## Data and methods

| Task | Inputs | Methods | Evaluation |
| --- | --- | --- | --- |
| Rating prediction | User IDs, book IDs, and ratings | Global-mean baseline, regularized user/book biases, and latent-factor model | Mean squared error on a random 90/10 training/validation split |
| Genre classification | Review text and genre labels | TF-IDF features and multiclass Logistic Regression | Accuracy on a random 80/20 training/validation split |

The five genre classes are children, comics/graphic, fantasy/paranormal, mystery/thriller/crime, and young adult.

## Modeling decisions

**Start with a simple rating baseline.** A global mean provides a reference for checking whether personalization adds predictive value. User and book biases account for differences in how users rate and how books are received.

**Explore latent factors with regularization and early stopping.** The latent-factor implementation models user–book interactions alongside biases, monitors validation error, and restores the best validation snapshot. Unknown users or books fall back to available bias information and zero latent factors.

**Fit text features on the training split.** The genre model learns its TF-IDF vocabulary from training reviews and transforms validation reviews separately. It evaluates five values of Logistic Regression's regularization parameter before refitting on all training reviews.

## Outputs

The script generates rating and genre prediction CSV files. It prints validation MSE and classification accuracy during execution; numerical performance results are not included here because saved evaluation outputs are not available in the reviewed source.

## Product implications

Rating prediction could support preference-based discovery, while genre classification could support content organization. These are possible applications of the models, rather than demonstrated product outcomes. A discovery system would also need ranking evaluation, coverage checks, and user testing.

## Limitations and next steps

- Random interaction splits do not establish performance on future behavior or entirely new users.
- Sparse histories limit personalization for new users and books.
- Report per-genre precision, recall, and confusion patterns alongside overall accuracy.
- Use a shared split and consistent evaluation rules when comparing rating models.
- Separate the active rating experiments: the current script writes `predictions_Rating.csv` twice, so its later bias-only model replaces the earlier latent-factor output.

## Tools and reproduction

Python, NumPy, scikit-learn, TF-IDF, Logistic Regression, and collaborative filtering.

The script expects `train_Interactions.csv.gz`, `pairs_Rating.csv`, `train_Category.json.gz`, and `test_Category.json.gz` in the working directory. With those inputs and the required packages available, run `python Book_reccomender.py`. This is an experimental script; execution was not validated as part of this documentation review.
