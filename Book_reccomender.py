import gzip
from collections import defaultdict
import string
from sklearn import linear_model
import math
import numpy as np
import random
from scipy.sparse import lil_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def readGz(path):
  for l in gzip.open(path, 'rt'):
    yield eval(l)

def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u, b, r = l.strip().split(',')
        r = int(r)
        yield u, b, r

### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before

# allRatings = []
# userRatings = defaultdict(list)

# for user,book,r in readCSV("train_Interactions.csv.gz"):
#   r = int(r)
#   allRatings.append(r)
#   userRatings[user].append(r)

# globalAverage = sum(allRatings) / len(allRatings)
# userAverage = {}
# for u in userRatings:
#   userAverage[u] = sum(userRatings[u]) / len(userRatings[u])

# predictions = open("predictions_Rating.csv", 'w')
# for l in open("pairs_Rating.csv"):
#   if l.startswith("userID"):
#     #header
#     predictions.write(l)
#     continue
#   u,b = l.strip().split(',')
#   if u in userAverage:
#     predictions.write(u + ',' + b + ',' + str(userAverage[u]) + '\n')
#   else:
#     predictions.write(u + ',' + b + ',' + str(globalAverage) + '\n')

# predictions.close()

### Rating prediction: alpha + betaU + betaI model (from HW3)

# ratings = []
# ratingsPerUser = defaultdict(list)
# ratingsPerItem = defaultdict(list)

# # Load training ratings
# for u,b,r in readCSV("train_Interactions.csv.gz"):
#     ratings.append((u,b,r))
#     ratingsPerUser[u].append((b,r))
#     ratingsPerItem[b].append((u,r))

# # Initialize parameters
# alpha = sum(r for _,_,r in ratings) / len(ratings)  # global mean
# betaU = {u: 0.0 for u in ratingsPerUser}           # user biases
# betaI = {b: 0.0 for b in ratingsPerItem}           # item biases

# lamb = 1.0
# T = 10  # number of iterations

# for t in range(T):
#     # --- update alpha ---
#     residuals = [r - (betaU[u] + betaI[b]) for (u,b,r) in ratings]
#     alpha = sum(residuals) / len(residuals)

#     # --- update betaU ---
#     newBetaU = {}
#     for u, ubRatings in ratingsPerUser.items():   # ubRatings: list of (b, r)
#         numer = sum(r - (alpha + betaI[b]) for (b,r) in ubRatings)
#         denom = lamb + len(ubRatings)
#         newBetaU[u] = numer / denom
#     betaU = newBetaU

#     # --- update betaI ---
#     newBetaI = {}
#     for b, brRatings in ratingsPerItem.items():   # brRatings: list of (u, r)
#         numer = sum(r - (alpha + betaU[u]) for (u,r) in brRatings)
#         denom = lamb + len(brRatings)
#         newBetaI[b] = numer / denom
#         # if you want, guard denom>0 but here it's always >0 for seen items
#     betaI = newBetaI

# predictions = open("predictions_Rating.csv", 'w')
# for l in open("pairs_Rating.csv"):
#     if l.startswith("userID"):
#         predictions.write(l)
#         continue
#     u,b = l.strip().split(',')
#     bu = betaU[u] if u in betaU else 0.0
#     bi = betaI[b] if b in betaI else 0.0
#     pred = alpha + bu + bi
#     predictions.write(u + ',' + b + ',' + str(pred) + '\n')

# predictions.close()

### Rating prediction: alpha + betaU + betaI with 90/10 validation (HW3 style)

# ---- Load all ratings from training set ----
# allRatings = []
# for u,b,r in readCSV("train_Interactions.csv.gz"):
#     allRatings.append((u,b,r))

# # ---- 90/10 train/validation split ----
# random.shuffle(allRatings)
# splitPoint = int(0.9 * len(allRatings))
# ratingsTrain = allRatings[:splitPoint]
# ratingsValid = allRatings[splitPoint:]

# # ---- Build per-user and per-item dicts on TRAIN only ----
# ratingsPerUser = defaultdict(list)
# ratingsPerItem = defaultdict(list)
# for u,b,r in ratingsTrain:
#     ratingsPerUser[u].append((b,r))
#     ratingsPerItem[b].append((u,r))

# # ---- Initialize parameters on TRAIN ----
# alpha = sum(r for _,_,r in ratingsTrain) / len(ratingsTrain)
# betaU = {u: 0.0 for u in ratingsPerUser}
# betaI = {b: 0.0 for b in ratingsPerItem}

# # Hyperparameters you can tune
# lamb = 5.0    # try 1.0, 3.0, 5.0, 10.0
# T = 15        # try 10, 15, 20

# # ---- Train on TRAIN split ----
# for t in range(T):
#     # Update alpha
#     residuals = [r - (betaU[u] + betaI[b]) for (u,b,r) in ratingsTrain]
#     alpha = sum(residuals) / len(residuals)

#     # Update betaU
#     newBetaU = {}
#     for u, ubRatings in ratingsPerUser.items():
#         numer = sum(r - (alpha + betaI[b]) for (b,r) in ubRatings)
#         denom = lamb + len(ubRatings)
#         newBetaU[u] = numer / denom
#     betaU = newBetaU

#     # Update betaI
#     newBetaI = {}
#     for b, brRatings in ratingsPerItem.items():
#         numer = sum(r - (alpha + betaU[u]) for (u,r) in brRatings)
#         denom = lamb + len(brRatings)
#         newBetaI[b] = numer / denom
#     betaI = newBetaI

# # ---- Compute validation MSE ----
# squaredErrors = []
# for u,b,r in ratingsValid:
#     bu = betaU[u] if u in betaU else 0.0
#     bi = betaI[b] if b in betaI else 0.0
#     pred = alpha + bu + bi
#     squaredErrors.append((r - pred) ** 2)
# validMSE = sum(squaredErrors) / len(squaredErrors)
# print("Validation MSE (90/10 split):", validMSE)

# # ------------------------------------------------------------------
# # After you’re happy with the validation MSE, retrain on ALL data
# # with the same lamb and T so the final model uses every rating.
# # ------------------------------------------------------------------

# ratingsPerUser_full = defaultdict(list)
# ratingsPerItem_full = defaultdict(list)
# for u,b,r in allRatings:
#     ratingsPerUser_full[u].append((b,r))
#     ratingsPerItem_full[b].append((u,r))

# alpha = sum(r for _,_,r in allRatings) / len(allRatings)
# betaU = {u: 0.0 for u in ratingsPerUser_full}
# betaI = {b: 0.0 for b in ratingsPerItem_full}

# for t in range(T):
#     residuals = [r - (betaU[u] + betaI[b]) for (u,b,r) in allRatings]
#     alpha = sum(residuals) / len(residuals)

#     newBetaU = {}
#     for u, ubRatings in ratingsPerUser_full.items():
#         numer = sum(r - (alpha + betaI[b]) for (b,r) in ubRatings)
#         denom = lamb + len(ubRatings)
#         newBetaU[u] = numer / denom
#     betaU = newBetaU

#     newBetaI = {}
#     for b, brRatings in ratingsPerItem_full.items():
#         numer = sum(r - (alpha + betaU[u]) for (u,r) in brRatings)
#         denom = lamb + len(brRatings)
#         newBetaI[b] = numer / denom
#     betaI = newBetaI

# # ---- Write predictions for the test pairs ----
# predictions = open("predictions_Rating.csv", 'w')
# for l in open("pairs_Rating.csv"):
#     if l.startswith("userID"):
#         predictions.write(l)
#         continue
#     u,b = l.strip().split(',')
#     bu = betaU[u] if u in betaU else 0.0
#     bi = betaI[b] if b in betaI else 0.0
#     pred = alpha + bu + bi
#     predictions.write(u + ',' + b + ',' + str(pred) + '\n')

# predictions.close()

#########################################################
# Q1: Rating prediction – latent factors + early stopping
#########################################################

# ---- Load all ratings ----
# allRatings = []
# for u, b, r in readCSV("train_Interactions.csv.gz"):
#     r = float(r)
#     allRatings.append((u, b, r))

# # ---- 90/10 train/validation split ----
# random.shuffle(allRatings)
# splitPoint = int(0.9 * len(allRatings))
# ratingsTrain = allRatings[:splitPoint]
# ratingsValid = allRatings[splitPoint:]

# # ---- Sanity check: trivial global-mean baseline on this split ----
# globalMean = sum(r for _, _, r in ratingsTrain) / len(ratingsTrain)
# sqErr_base = [(r - globalMean) ** 2 for (_, _, r) in ratingsValid]
# baseMSE = sum(sqErr_base) / len(sqErr_base)
# print(f"Baseline (global mean) validation MSE: {baseMSE:.6f}")

# # ---- Collect users/items from TRAIN only ----
# usersTrain = {u for (u, _, _) in ratingsTrain}
# itemsTrain = {b for (_, b, _) in ratingsTrain}

# # ---- Initialize bias terms ----
# alpha = globalMean
# betaU = {u: 0.0 for u in usersTrain}
# betaI = {b: 0.0 for b in itemsTrain}

# # ---- Initialize latent factors (k = 1) ----
# k = 1
# userFactors = {u: numpy.random.normal(0, 0.1, k) for u in usersTrain}
# itemFactors = {b: numpy.random.normal(0, 0.1, k) for b in itemsTrain}

# # ---- Hyperparameters ----
# lrate     = 0.005     # learning rate
# regBias   = 0.01     # bias regularization
# regLatent = 0.01     # latent factor regularization
# maxEpochs = 30
# patience  = 3

# bestValMSE      = float('inf')
# bestEpoch       = 0
# bestAlpha       = alpha
# bestBetaU       = dict(betaU)
# bestBetaI       = dict(betaI)
# bestUserFactors = {u: f.copy() for u, f in userFactors.items()}
# bestItemFactors = {b: f.copy() for b, f in itemFactors.items()}
# noImprove       = 0

# # ---- SGD training with early stopping ----
# for epoch in range(1, maxEpochs + 1):
#     random.shuffle(ratingsTrain)

#     for u, b, r in ratingsTrain:
#         pu = userFactors.get(u)
#         qi = itemFactors.get(b)
#         # if pu is None:
#         #     pu = numpy.zeros(k)
#         #     userFactors[u] = pu
#         #     betaU[u] = 0.0
#         # if qi is None:
#         #     qi = numpy.zeros(k)
#         #     itemFactors[b] = qi
#         #     betaI[b] = 0.0

#         bu = betaU[u]
#         bi = betaI[b]

#         # prediction *without* clamping
#         pred = alpha + bu + bi + numpy.dot(pu, qi)
#         err  = r - pred

#         # update global bias
#         alpha += lrate * err

#         # update user/item biases
#         betaU[u] += lrate * (err - regBias * bu)
#         betaI[b] += lrate * (err - regBias * bi)

#         # update latent factors
#         pu_old = pu.copy()
#         qi_old = qi.copy()
#         userFactors[u] += lrate * (err * qi_old - regLatent * pu_old)
#         itemFactors[b] += lrate * (err * pu_old - regLatent * qi_old)

#     # ---- compute validation MSE (no clamping) ----
#     sqErr = []
#     for u, b, r in ratingsValid:
#         bu = betaU.get(u, 0.0)
#         bi = betaI.get(b, 0.0)
#         pu = userFactors.get(u, numpy.zeros(k))
#         qi = itemFactors.get(b, numpy.zeros(k))
#         pred = alpha + bu + bi + numpy.dot(pu, qi)
#         sqErr.append((r - pred) ** 2)

#     valMSE = sum(sqErr) / len(sqErr)
#     print(f"Epoch {epoch}, validation MSE: {valMSE:.6f}")

#     # ---- early stopping bookkeeping ----
#     if valMSE < bestValMSE - 1e-6:
#         bestValMSE      = valMSE
#         bestEpoch       = epoch
#         bestAlpha       = alpha
#         bestBetaU       = dict(betaU)
#         bestBetaI       = dict(betaI)
#         bestUserFactors = {u: f.copy() for u, f in userFactors.items()}
#         bestItemFactors = {b: f.copy() for b, f in itemFactors.items()}
#         noImprove       = 0
#     else:
#         noImprove += 1
#         if noImprove >= patience:
#             print(f"Early stopping after epoch {epoch}, best epoch = {bestEpoch}")
#             break

# print("Best validation MSE:", bestValMSE, "at epoch", bestEpoch)


# # ==========================================
# # Final model: retrain on ALL data
# # using bestEpoch and same hyperparams
# # ==========================================

# usersAll = {u for (u, _, _) in allRatings}
# itemsAll = {b for (_, b, _) in allRatings}

# alpha = sum(r for _, _, r in allRatings) / len(allRatings)
# betaU = {u: 0.0 for u in usersAll}
# betaI = {b: 0.0 for b in itemsAll}
# userFactors = {u: numpy.random.normal(0, 0.1, k) for u in usersAll}
# itemFactors = {b: numpy.random.normal(0, 0.1, k) for b in itemsAll}

# for epoch in range(bestEpoch):
#     random.shuffle(allRatings)
#     for u, b, r in allRatings:
#         pu = userFactors[u]
#         qi = itemFactors[b]
#         bu = betaU[u]
#         bi = betaI[b]

#         pred = alpha + bu + bi + numpy.dot(pu, qi)
#         err  = r - pred

#         alpha     += lrate * err
#         betaU[u]  += lrate * (err - regBias * bu)
#         betaI[b]  += lrate * (err - regBias * bi)

#         pu_old = pu.copy()
#         qi_old = qi.copy()
#         userFactors[u] += lrate * (err * qi_old - regLatent * pu_old)
#         itemFactors[b] += lrate * (err * pu_old - regLatent * qi_old)


# # ==========================================
# # Final model for predictions on ALL data
# # (you can either retrain or just reuse
# # bestAlpha/betaU/betaI/userFactors/itemFactors;
# # using the best snapshot is fine)
# # ==========================================

# alpha       = bestAlpha
# betaU       = bestBetaU
# betaI       = bestBetaI
# userFactors = bestUserFactors
# itemFactors = bestItemFactors

#########################################################
# Q1: Rating prediction – latent factors + early stopping
#########################################################

# ---- Load all ratings ----
allRatings = []
for u, b, r in readCSV("train_Interactions.csv.gz"):
    r = float(r)
    allRatings.append((u, b, r))

# ---- 90/10 train/validation split ----
random.shuffle(allRatings)
splitPoint   = int(0.9 * len(allRatings))
ratingsTrain = allRatings[:splitPoint]
ratingsValid = allRatings[splitPoint:]

# ---- Baseline sanity check ----
globalMean   = sum(r for _, _, r in ratingsTrain) / len(ratingsTrain)
sqErr_base   = [(r - globalMean) ** 2 for (_, _, r) in ratingsValid]
baseMSE      = sum(sqErr_base) / len(sqErr_base)
print(f"Baseline (global mean) validation MSE: {baseMSE:.6f}")

# ---- Collect users/items from TRAIN only ----
usersTrain = {u for (u, _, _) in ratingsTrain}
itemsTrain = {b for (_, b, _) in ratingsTrain}

# ---- Initialize parameters ----
alpha = globalMean
betaU = {u: 0.0 for u in usersTrain}
betaI = {b: 0.0 for b in itemsTrain}

k = 1  # latent dimension
userFactors = {u: np.random.normal(0, 0.1, k) for u in usersTrain}
itemFactors = {b: np.random.normal(0, 0.1, k) for b in itemsTrain}

# ---- Hyperparameters (more conservative) ----
lrate     = 0.003    # smaller step size
regBias   = 0.02     # stronger bias regularization
regLatent = 0.05     # stronger latent regularization
maxEpochs = 25
patience  = 2

bestValMSE      = float('inf')
bestEpoch       = 0
bestAlpha       = alpha
bestBetaU       = dict(betaU)
bestBetaI       = dict(betaI)
bestUserFactors = {u: f.copy() for u, f in userFactors.items()}
bestItemFactors = {b: f.copy() for b, f in itemFactors.items()}
noImprove       = 0

# ---- SGD training with early stopping ----
for epoch in range(1, maxEpochs + 1):
    random.shuffle(ratingsTrain)

    for u, b, r in ratingsTrain:
        pu = userFactors[u]
        qi = itemFactors[b]
        bu = betaU[u]
        bi = betaI[b]

        pred = alpha + bu + bi + np.dot(pu, qi)
        err  = r - pred

        # update global bias
        alpha += lrate * err

        # update user/item biases
        betaU[u] += lrate * (err - regBias * bu)
        betaI[b] += lrate * (err - regBias * bi)

        # update latent factors
        pu_old = pu.copy()
        qi_old = qi.copy()
        userFactors[u] += lrate * (err * qi_old - regLatent * pu_old)
        itemFactors[b] += lrate * (err * pu_old - regLatent * qi_old)

    # ---- validation MSE ----
    sqErr = []
    for u, b, r in ratingsValid:
        bu = betaU.get(u, 0.0)
        bi = betaI.get(b, 0.0)
        pu = userFactors.get(u, np.zeros(k))
        qi = itemFactors.get(b, np.zeros(k))
        pred = alpha + bu + bi + np.dot(pu, qi)
        sqErr.append((r - pred) ** 2)

    valMSE = sum(sqErr) / len(sqErr)
    print(f"Epoch {epoch}, validation MSE: {valMSE:.6f}")

    # ---- early stopping ----
    if valMSE < bestValMSE - 1e-6:
        bestValMSE      = valMSE
        bestEpoch       = epoch
        bestAlpha       = alpha
        bestBetaU       = dict(betaU)
        bestBetaI       = dict(betaI)
        bestUserFactors = {u: f.copy() for u, f in userFactors.items()}
        bestItemFactors = {b: f.copy() for b, f in itemFactors.items()}
        noImprove       = 0
    else:
        noImprove += 1
        if noImprove >= patience:
            print(f"Early stopping after epoch {epoch}, best epoch = {bestEpoch}")
            break

print("Best validation MSE:", bestValMSE, "at epoch", bestEpoch)

# ==========================================
# Final model = best snapshot (no retrain)
# ==========================================

alpha       = bestAlpha
betaU       = bestBetaU
betaI       = bestBetaI
userFactors = bestUserFactors
itemFactors = bestItemFactors

# ---- Predictions (with clamping only here) ----
predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue

    u, b = l.strip().split(',')
    bu = betaU.get(u, 0.0)
    bi = betaI.get(b, 0.0)
    pu = userFactors.get(u, np.zeros(k))
    qi = itemFactors.get(b, np.zeros(k))

    pred = alpha + bu + bi + np.dot(pu, qi)
    pred = max(1.0, min(5.0, pred))
    predictions.write(u + ',' + b + ',' + str(pred) + '\n')

predictions.close()





### Would-read baseline: just rank which books are popular and which are not, and return '1' if a book is among the top-ranked

# bookCount = defaultdict(int)
# totalRead = 0

# for user,book,_ in readCSV("train_Interactions.csv.gz"):
#   bookCount[book] += 1
#   totalRead += 1

# mostPopular = [(bookCount[x], x) for x in bookCount]
# mostPopular.sort()
# mostPopular.reverse()

# return1 = set()
# count = 0
# for ic, i in mostPopular:
#   count += ic
#   return1.add(i)
#   if count > totalRead/2: break

# predictions = open("predictions_Read.csv", 'w')
# for l in open("pairs_Read.csv"):
#   if l.startswith("userID"):
#     #header
#     predictions.write(l)
#     continue
#   u,b = l.strip().split(',')
#   if b in return1:
#     predictions.write(u + ',' + b + ",1\n")
#   else:
#     predictions.write(u + ',' + b + ",0\n")

# predictions.close()

### Improved would-read model: popularity with tuned cutoff (like HW3 improvedStrategy)

# bookCount = defaultdict(int)
# totalRead = 0

# for user,book,_ in readCSV("train_Interactions.csv.gz"):
#   bookCount[book] += 1
#   totalRead += 1

# mostPopular = [(bookCount[x], x) for x in bookCount]
# mostPopular.sort()
# mostPopular.reverse()

# return1 = set()
# count = 0
# cutoff = totalRead * 0.75   # more conservative than 0.5, like your HW3 improvedStrategy

# for ic, i in mostPopular:
#   count += ic
#   return1.add(i)
#   if count > cutoff:
#     break

# predictions = open("predictions_Read.csv", 'w')
# for l in open("pairs_Read.csv"):
#   if l.startswith("userID"):
#     # header
#     predictions.write(l)
#     continue
#   u,b = l.strip().split(',')
#   if b in return1:
#     predictions.write(u + ',' + b + ",1\n")
#   else:
#     predictions.write(u + ',' + b + ",0\n")

# predictions.close()

### Improved would-read model: Jaccard + popularity (HW3-style, non-trivial)

# If you've already built ratingsPerUser and ratingsPerItem above for rating,
# you can reuse them. If not, build them here:

# try:
#     ratingsPerUser
#     ratingsPerItem
# except NameError:
#     ratingsPerUser = defaultdict(list)
#     ratingsPerItem = defaultdict(list)
#     for u, b, r in readCSV("train_Interactions.csv.gz"):
#         ratingsPerUser[u].append((b, r))
#         ratingsPerItem[b].append((u, r))

# # Precompute: for each book, the set of users who read it
# usersForItem = {}
# for b, brRatings in ratingsPerItem.items():
#     usersForItem[b] = set([u for (u, r) in brRatings])

# # Also precompute simple popularity (how many users read each book)
# bookPopularity = {b: len(usersForItem[b]) for b in usersForItem}

# # Tunable thresholds (you can adjust these based on a 90/10 validation split)
# simThresh = 0.013   # from HW3 jaccardThresh
# popThresh = 40      # also similar to HW3 idea

# def jaccard_sim(s1, s2):
#     if not s1 or not s2:
#         return 0.0
#     inter = len(s1.intersection(s2))
#     union = len(s1.union(s2))
#     if union == 0:
#         return 0.0
#     return inter / union

# predictions = open("predictions_Read.csv", 'w')
# for l in open("pairs_Read.csv"):
#     if l.startswith("userID"):
#         predictions.write(l)
#         continue

#     u, b = l.strip().split(',')

#     # Default: not read
#     label = 0

#     # If we've never seen the item, fall back to popularity-based guess=0
#     if b in usersForItem:
#         users_b = usersForItem[b]

#         # If user has history, compute max Jaccard similarity to their items
#         maxSim = 0.0
#         if u in ratingsPerUser:
#             for b2, _ in ratingsPerUser[u]:
#                 if b2 not in usersForItem:
#                     continue
#                 sim = jaccard_sim(users_b, usersForItem[b2])
#                 if sim > maxSim:
#                     maxSim = sim

#         # Popularity of the target item
#         pop = bookPopularity.get(b, 0)

#         # Decision rule: similar-to-history OR popular-enough
#         if maxSim > simThresh or pop > popThresh:
#             label = 1

#     predictions.write(u + ',' + b + ',' + str(label) + '\n')

# predictions.close()

# #########################################################
# # Q1: Rating prediction – latent factors (k=1)          #
# #      conservative hyperparams + early stopping        #
# #########################################################

# # For reproducibility (so your local numbers don't jump as much)

# # ---- Load all ratings ----
# allRatings = []
# for u, b, r in readCSV("train_Interactions.csv.gz"):
#     r = float(r)
#     allRatings.append((u, b, r))

# # ---- 90/10 train/validation split ----
# random.shuffle(allRatings)
# splitPoint   = int(0.9 * len(allRatings))
# ratingsTrain = allRatings[:splitPoint]
# ratingsValid = allRatings[splitPoint:]

# # ---- Simple baseline on this split (optional sanity check) ----
# globalMean  = sum(r for _, _, r in ratingsTrain) / len(ratingsTrain)
# sqErr_base  = [(r - globalMean) ** 2 for (_, _, r) in ratingsValid]
# baseMSE     = sum(sqErr_base) / len(sqErr_base)
# print(f"Baseline (global mean) validation MSE: {baseMSE:.6f}")

# # ---- Users/items from TRAIN ----
# usersTrain = {u for (u, _, _) in ratingsTrain}
# itemsTrain = {b for (_, b, _) in ratingsTrain}

# # ---- Initialize biases + latent factors ----
# alpha = globalMean
# betaU = {u: 0.0 for u in usersTrain}
# betaI = {b: 0.0 for b in itemsTrain}

# k = 1  # latent dimension
# userFactors = {u: numpy.random.normal(0, 0.1, k) for u in usersTrain}
# itemFactors = {b: numpy.random.normal(0, 0.1, k) for b in itemsTrain}

# # ---- Conservative hyperparameters (Option B) ----
# lrate     = 0.008   # learning rate
# regBias   = 0.005   # L2 on biases
# regLatent = 0.01    # L2 on latent factors
# maxEpochs = 25
# patience  = 3       # stop if val MSE stops improving

# bestValMSE = float('inf')
# bestEpoch  = 0
# bestAlpha  = alpha
# bestBetaU  = dict(betaU)
# bestBetaI  = dict(betaI)
# bestUserF  = {u: f.copy() for u, f in userFactors.items()}
# bestItemF  = {b: f.copy() for b, f in itemFactors.items()}
# noImprove  = 0

# # ---- SGD training with early stopping ----
# for epoch in range(1, maxEpochs + 1):
#     random.shuffle(ratingsTrain)

#     for u, b, r in ratingsTrain:
#         pu = userFactors[u]
#         qi = itemFactors[b]
#         bu = betaU[u]
#         bi = betaI[b]

#         # prediction (no clamping during training)
#         pred = alpha + bu + bi + numpy.dot(pu, qi)
#         err  = r - pred

#         # update global bias
#         alpha += lrate * err

#         # update user/item biases
#         betaU[u] += lrate * (err - regBias * bu)
#         betaI[b] += lrate * (err - regBias * bi)

#         # update latent factors
#         pu_old = pu.copy()
#         qi_old = qi.copy()
#         userFactors[u] += lrate * (err * qi_old - regLatent * pu_old)
#         itemFactors[b] += lrate * (err * pu_old - regLatent * qi_old)

#     # ---- validation MSE ----
#     sqErr = []
#     for u, b, r in ratingsValid:
#         bu = betaU.get(u, 0.0)
#         bi = betaI.get(b, 0.0)
#         pu = userFactors.get(u, numpy.zeros(k))
#         qi = itemFactors.get(b, numpy.zeros(k))
#         pred = alpha + bu + bi + numpy.dot(pu, qi)
#         sqErr.append((r - pred) ** 2)

#     valMSE = sum(sqErr) / len(sqErr)
#     print(f"Epoch {epoch}, validation MSE: {valMSE:.6f}")

#     # ---- early stopping bookkeeping ----
#     if valMSE < bestValMSE - 1e-6:
#         bestValMSE = valMSE
#         bestEpoch  = epoch
#         bestAlpha  = alpha
#         bestBetaU  = dict(betaU)
#         bestBetaI  = dict(betaI)
#         bestUserF  = {u: f.copy() for u, f in userFactors.items()}
#         bestItemF  = {b: f.copy() for b, f in itemFactors.items()}
#         noImprove  = 0
#     else:
#         noImprove += 1
#         if noImprove >= patience:
#             print(f"Early stopping after epoch {epoch}, best epoch = {bestEpoch}")
#             break

# print("Best validation MSE:", bestValMSE, "at epoch", bestEpoch)

# # ---- Use best snapshot directly for predictions ----
# alpha       = bestAlpha
# betaU       = bestBetaU
# betaI       = bestBetaI
# userFactors = bestUserF
# itemFactors = bestItemF

# # ---- Write predictions (clamp ONLY here) ----
# predictions = open("predictions_Rating.csv", 'w')
# for l in open("pairs_Rating.csv"):
#     if l.startswith("userID"):
#         predictions.write(l)
#         continue

#     u, b = l.strip().split(',')
#     bu = betaU.get(u, 0.0)
#     bi = betaI.get(b, 0.0)
#     pu = userFactors.get(u, numpy.zeros(k))
#     qi = itemFactors.get(b, numpy.zeros(k))

#     pred = alpha + bu + bi + numpy.dot(pu, qi)
#     pred = max(1.0, min(5.0, pred))  # clamp only for final output
#     predictions.write(u + ',' + b + ',' + str(pred) + '\n')

# predictions.close()

################################################
# Q1: Rating prediction – bias-only model      #
#################################################

# ---- Load all ratings ----
allRatings = []
for u, b, r in readCSV("train_Interactions.csv.gz"):
    r = float(r)
    allRatings.append((u, b, r))

# ---- 90/10 train/validation split ----
random.shuffle(allRatings)
splitPoint   = int(0.9 * len(allRatings))
ratingsTrain = allRatings[:splitPoint]
ratingsValid = allRatings[splitPoint:]

# ---- Build per-user and per-item dicts on TRAIN ----
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u, b, r in ratingsTrain:
    ratingsPerUser[u].append((b, r))
    ratingsPerItem[b].append((u, r))

# ---- Initialize parameters on TRAIN ----
alpha = sum(r for _, _, r in ratingsTrain) / len(ratingsTrain)
betaU = {u: 0.0 for u in ratingsPerUser}
betaI = {b: 0.0 for b in ratingsPerItem}

# ---- Hyperparameters ----
lamb = 5.0          # regularizer
Tmax = 30           # maximum number of iterations to try

bestValidMSE = float('inf')
bestIter     = 0
bestAlpha    = alpha
bestBetaU    = dict(betaU)
bestBetaI    = dict(betaI)

# ---- Coordinate descent on TRAIN, monitor VALID ----
for t in range(1, Tmax + 1):
    # Update alpha
    residuals = [r - (betaU[u] + betaI[b]) for (u, b, r) in ratingsTrain]
    alpha = sum(residuals) / len(residuals)

    # Update betaU
    newBetaU = {}
    for u, ubRatings in ratingsPerUser.items():
        numer = sum(r - (alpha + betaI[b]) for (b, r) in ubRatings)
        denom = lamb + len(ubRatings)
        newBetaU[u] = numer / denom
    betaU = newBetaU

    # Update betaI
    newBetaI = {}
    for b, brRatings in ratingsPerItem.items():
        numer = sum(r - (alpha + betaU[u]) for (u, r) in brRatings)
        denom = lamb + len(brRatings)
        newBetaI[b] = numer / denom
    betaI = newBetaI

    # ---- Compute validation MSE (with clamping) ----
    sqErr = []
    for u, b, r in ratingsValid:
        bu   = betaU.get(u, 0.0)
        bi   = betaI.get(b, 0.0)
        pred = alpha + bu + bi
        pred = max(1.0, min(5.0, pred))   # clamp to [1,5]
        sqErr.append((r - pred) ** 2)
    validMSE = sum(sqErr) / len(sqErr)
    print(f"Iter {t}, validation MSE: {validMSE}")

    # Track best iteration
    if validMSE < bestValidMSE:
        bestValidMSE = validMSE
        bestIter     = t
        bestAlpha    = alpha
        bestBetaU    = dict(betaU)
        bestBetaI    = dict(betaI)

print("Best validation MSE:", bestValidMSE, "at iteration", bestIter)

# ======================================================
# Retrain on ALL data for bestIter iterations
# ======================================================

# Rebuild per-user/per-item on ALL data
ratingsPerUser_full = defaultdict(list)
ratingsPerItem_full = defaultdict(list)
for u, b, r in allRatings:
    ratingsPerUser_full[u].append((b, r))
    ratingsPerItem_full[b].append((u, r))

alpha = sum(r for _, _, r in allRatings) / len(allRatings)
betaU = {u: 0.0 for u in ratingsPerUser_full}
betaI = {b: 0.0 for b in ratingsPerItem_full}

for t in range(bestIter):
    # Update alpha
    residuals = [r - (betaU[u] + betaI[b]) for (u, b, r) in allRatings]
    alpha = sum(residuals) / len(residuals)

    # Update betaU
    newBetaU = {}
    for u, ubRatings in ratingsPerUser_full.items():
        numer = sum(r - (alpha + betaI[b]) for (b, r) in ubRatings)
        denom = lamb + len(ubRatings)
        newBetaU[u] = numer / denom
    betaU = newBetaU

    # Update betaI
    newBetaI = {}
    for b, brRatings in ratingsPerItem_full.items():
        numer = sum(r - (alpha + betaU[u]) for (u, r) in brRatings)
        denom = lamb + len(brRatings)
        newBetaI[b] = numer / denom
    betaI = newBetaI

# ---- Final predictions on pairs_Rating.csv ----
predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue

    u, b = l.strip().split(',')
    bu   = betaU.get(u, 0.0)
    bi   = betaI.get(b, 0.0)
    pred = alpha + bu + bi
    pred = max(1.0, min(5.0, pred))    # clamp ONLY here for final output
    predictions.write(u + ',' + b + ',' + str(pred) + '\n')

predictions.close()



### Category prediction baseline: Just consider some of the most common words from each category

# catDict = {
#   "children": 0,
#   "comics_graphic": 1,
#   "fantasy_paranormal": 2,
#   "mystery_thriller_crime": 3,
#   "young_adult": 4
# }

# predictions = open("predictions_Category.csv", 'w')
# predictions.write("userID,reviewID,prediction\n")
# for l in readGz("test_Category.json.gz"):
#   cat = catDict['fantasy_paranormal'] # If there's no evidence, just choose the most common category in the dataset
#   words = l['review_text'].lower()
#   if 'children' in words:
#     cat = catDict['children']
#   if 'comic' in words:
#     cat = catDict['comics_graphic']
#   if 'fantasy' in words:
#     cat = catDict['fantasy_paranormal']
#   if 'mystery' in words:
#     cat = catDict['mystery_thriller_crime']
#   if 'love' in words:
#     cat = catDict['young_adult']
#   predictions.write(l['user_id'] + ',' + l['review_id'] + "," + str(cat) + "\n")

# predictions.close()

### Improved Category prediction: bag-of-words + logistic regression (HW3 style)

# Map is still useful conceptually but we don't actually need it for training,
# since 'genreID' is already numeric:
# catDict = {
#   "children": 0,
#   "comics_graphic": 1,
#   "fantasy_paranormal": 2,
#   "mystery_thriller_crime": 3,
#   "young_adult": 4
# }

# # --- Load training data for category prediction ---
# trainData = []
# for d in readGz("train_Category.json.gz"):
#   trainData.append(d)

# # --- Build vocabulary from training reviews ---
# punct = string.punctuation
# wordCount = defaultdict(int)

# for d in trainData:
#   t = d['review_text'].lower()
#   t = ''.join([c for c in t if c not in punct])
#   for w in t.split():
#     wordCount[w] += 1

# # Choose top-K frequent words as features (tune K if needed)
# K = 5000  # you can change this to 2000 / 10000 later if you want to tune

# sortedWords = sorted(wordCount.items(), key=lambda x: -x[1])
# words = [w for (w, _) in sortedWords[:K]]
# wordId = {w:i for (i,w) in enumerate(words)}
# wordSet = set(words)

# # --- Build feature matrix X and label vector y ---
# X = []
# y = []

# for d in trainData:
#   t = d['review_text'].lower()
#   t = ''.join([c for c in t if c not in punct])
#   toks = t.split()
#   feat = [0]*K
#   for w in toks:
#     if w in wordSet:
#       feat[wordId[w]] += 1
#   feat.append(1)  # bias term
#   X.append(feat)
#   y.append(d['genreID'])

# # --- Train logistic regression classifier ---
# logReg = linear_model.LogisticRegression(
#   max_iter=1000,
#   class_weight="balanced",
#   multi_class="auto"
# )
# logReg.fit(X, y)

# # --- Predict on test set and write predictions ---
# predictions = open("predictions_Category.csv", 'w')
# predictions.write("userID,reviewID,prediction\n")

# for d in readGz("test_Category.json.gz"):
#   t = d['review_text'].lower()
#   t = ''.join([c for c in t if c not in punct])
#   toks = t.split()
#   feat = [0]*K
#   for w in toks:
#     if w in wordSet:
#       feat[wordId[w]] += 1
#   feat.append(1)
#   pred = logReg.predict([feat])[0]
#   predictions.write(d['user_id'] + ',' + d['review_id'] + ',' + str(pred) + '\n')

# predictions.close()

catDict = {
    "children": 0,
    "comics_graphic": 1,
    "fantasy_paranormal": 2,
    "mystery_thriller_crime": 3,
    "young_adult": 4
}

# ----------- Load training data -----------
trainCatData = []
for d in readGz("train_Category.json.gz"):
    trainCatData.append(d)

Ntrain = len(trainCatData)
print("Num training reviews:", Ntrain)

# ----------- Build texts + labels ----------
texts = []
labels = []
for d in trainCatData:
    texts.append(d['review_text'].lower())
    labels.append(d['genreID'])

labels = np.array(labels)

# ====================================================
# Split into train/validation for hyperparameter tuning
# ====================================================
indices = list(range(Ntrain))
random.shuffle(indices)
splitPoint = int(0.8 * Ntrain)    # 80% train, 20% validation
train_idx = indices[:splitPoint]
valid_idx = indices[splitPoint:]

textsTrain = [texts[i] for i in train_idx]
textsValid = [texts[i] for i in valid_idx]
labelsTrain = labels[train_idx]
labelsValid = labels[valid_idx]

# ----------- TF-IDF Vectorizer (fit on train split) ------------
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    max_features=20000,   # big-ish vocab
    min_df=5,
    max_df=0.80,
    sublinear_tf=True,
)

Xtrain = vectorizer.fit_transform(textsTrain)
Xvalid = vectorizer.transform(textsValid)

# ----------- Grid search over C ------------
C_grid = [0.5, 1.0, 2.0, 3.0, 5.0]

bestC = None
bestValAcc = 0.0

for C in C_grid:
    clf = LogisticRegression(
        C=C,
        max_iter=1000  # enough to converge
        # default solver='lbfgs' works fine for multiclass
    )
    clf.fit(Xtrain, labelsTrain)
    valAcc = clf.score(Xvalid, labelsValid)
    print(f"[grid] C={C}, validation accuracy={valAcc:.4f}")

    if valAcc > bestValAcc:
        bestValAcc = valAcc
        bestC = C

print(f"Best C from grid search: {bestC} with validation accuracy={bestValAcc:.4f}")

# ====================================================
# Re-fit on ALL training data with best C
# ====================================================
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    max_features=20000,
    min_df=5,
    max_df=0.80,
    sublinear_tf=True,
)

X_all = vectorizer.fit_transform(texts)
y_all = labels

print("Training final classifier with best C...")
logRegCat = LogisticRegression(
    C=bestC,
    max_iter=1000
)
logRegCat.fit(X_all, y_all)

# ----------- Predict on test set ------------
print("Predicting on test set...")

predictions = open("predictions_Category.csv", "w")
predictions.write("userID,reviewID,prediction\n")

for d in readGz("test_Category.json.gz"):
    t = d['review_text'].lower()
    X = vectorizer.transform([t])
    pred = logRegCat.predict(X)[0]
    predictions.write(f"{d['user_id']},{d['review_id']},{pred}\n")

predictions.close()
print("Wrote predictions_Category.csv")