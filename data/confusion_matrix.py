import matplotlib.pyplot as plt
import numpy as np

cm = np.array([
    [434, 177],
    [104, 285]
])

classes = ["Stay", "Churn"]

plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens) # type: ignore

plt.title("Confusion Matrix", fontsize=18, pad=20)
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, fontsize=12)
plt.yticks(tick_marks, classes, fontsize=12)

plt.xlabel("Prediction", fontsize=14)
plt.ylabel("Actual", fontsize=14)

plt.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > 300 else "black",
            fontsize=14
        )

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()