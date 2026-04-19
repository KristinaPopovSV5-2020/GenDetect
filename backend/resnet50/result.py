import matplotlib.pyplot as plt


epochs = list(range(1, 21))

train_loss = [
    0.5531, 0.4389, 0.3876, 0.3689, 0.3477,
    0.3404, 0.3277, 0.3153, 0.3048, 0.3026,
    0.2916, 0.2810, 0.2808, 0.2727, 0.2689,
    0.2537, 0.2495, 0.2484, 0.2369, 0.2310
]

train_auc = [
    0.8112, 0.8813, 0.9084, 0.9163, 0.9259,
    0.9291, 0.9343, 0.9391, 0.9431, 0.9439,
    0.9480, 0.9518, 0.9517, 0.9545, 0.9557,
    0.9608, 0.9619, 0.9622, 0.9657, 0.9675
]

val_auc = [
    0.9027, 0.9263, 0.9373, 0.9407, 0.9434,
    0.9484, 0.9500, 0.9516, 0.9505, 0.9529,
    0.9533, 0.9557, 0.9545, 0.9584, 0.9584,
    0.9571, 0.9602, 0.9598, 0.9598, 0.9599
]


plt.figure()
plt.plot(epochs, train_auc, label="Train AUC")
plt.plot(epochs, val_auc, label="Val AUC")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.legend()
plt.title("Train vs Validation AUC")
plt.show()