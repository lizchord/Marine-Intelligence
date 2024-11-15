from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics
from matplotlib import pyplot as plt
from data_provider.preprocess import load_my_fancy_dataset

cpue, mfd = load_my_fancy_dataset()
X = mfd.data
Y = mfd.target

count_0 = 0
count_1 = 0
for i in range(Y.shape[0]):
    if Y[i] == 0:
        count_0 = count_0 + 1
    else:
        count_1 = count_1 + 1
print(f"Original imbalance ratios (count_0 / count_1) :", count_0/count_1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

LR = LogisticRegression(random_state=0)
LR.fit(x_train, y_train)

y_pred_baseline = LR.predict(x_test)
print(f"baseline1:")
print(classification_report(y_test, y_pred_baseline, digits=4))

y_pred_proba = LR.predict_proba(x_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba[:, 1], pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.4f)" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
# plt.savefig('auc_roc.pdf')
plt.show()


clf = LogisticRegression(random_state=0)
clf.fit(x_train, y_train)

num_crossval_folds = 5  # for efficiency; values like 5 or 10 will generally work better
pred_probs = cross_val_predict(
    clf,
    x_train,
    y_train,
    cv=num_crossval_folds,
    method="predict_proba",
)

predicted_labels = pred_probs.argmax(axis=1)

LR_2 = LogisticRegression(random_state=0)
LR_2.fit(x_train, predicted_labels)

y_pred_baseline = LR_2.predict(x_test)
print(f"baseline2:")
print(classification_report(y_test, y_pred_baseline, digits=4))

y_pred_proba_baseline_2 = LR_2.predict_proba(x_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba_baseline_2[:, 1], pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.4f)" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
# plt.savefig('auc_roc.pdf')
plt.show()

from core.cleanlab.dataset import overall_label_health_score

#global cleaning
clf = LogisticRegression(random_state=0)
pred_probs_global = cross_val_predict(
    clf,
    X,
    Y,
    cv=num_crossval_folds,
    method="predict_proba",
)

count_0 = 0
count_1 = 0
predicted_label_global = pred_probs_global.argmax(axis=1)

for i in range(predicted_label_global.shape[0]):
    if predicted_label_global[i] == 0:
        count_0 = count_0 + 1
    else:
        count_1 = count_1 + 1
print(f"Baseline2_imbalance_ratio (count_0 / count_1) :", count_0/count_1)

potential_noise_in_cpue_0 = 0
count_error = 0
for i in range(predicted_label_global.shape[0]):
    if predicted_label_global[i] != Y[i]:
        count_error = count_error + 1
        if cpue[i] == 0:
            potential_noise_in_cpue_0 = potential_noise_in_cpue_0 + 1

cpue_0_count = 0
for i in range(cpue.shape[0]):
    if cpue[i] == 0:
        cpue_0_count = cpue_0_count + 1

print(f"**********our approach**********")
health_global, error_indices_global = overall_label_health_score(Y, pred_probs_global)
for i in range(error_indices_global.shape[0]):
    Y[error_indices_global[i]] = not Y[error_indices_global[i]]

count_0 = 0
count_1 = 0
for i in range(Y.shape[0]):
    if Y[i] == 0:
        count_0 = count_0 + 1
    else:
        count_1 = count_1 + 1
print(f"Our_approach (count_0 / count_1) :", count_0/count_1)

x_train_global, x_test_global, y_train_global, y_test_global = train_test_split(X, Y, test_size=0.2)

ours = LogisticRegression(random_state=0)
ours.fit(x_train_global, y_train_global)

y_pred_global = ours.predict(x_test_global)
print(classification_report(y_test_global, y_pred_global, digits=4))

noise_0 = 0
noise_1 = 0

for i in range(error_indices_global.shape[0]):
    if cpue[error_indices_global[i]] == 0:
        noise_0 = noise_0 + 1
print(f"Z / D (raw proportion): {cpue_0_count}")
print(f"N / D: {error_indices_global.shape[0]/Y.shape[0]}")
print(f"N(Z) / N: {noise_0/error_indices_global.shape[0]}")
print(f"Z(N) / Z: {noise_0/cpue_0_count}")

y_pred_global = ours.predict_proba(x_test_global)
fpr, tpr, thresholds = metrics.roc_curve(y_test_global, y_pred_global[:, 1], pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.4f)" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
# plt.savefig('auc_roc.pdf')
plt.show()
