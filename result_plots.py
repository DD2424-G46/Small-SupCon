import matplotlib.pyplot as plt




def create_plot(x_data, y_datas, x_label, y_label, title, close=True, legends=None, ylim=None):
    if any(isinstance(y_data, list) for y_data in y_datas):
        for i, y_data in enumerate(y_datas):
            plt.plot(x_data, y_data, marker='o', label=legends[i] if legends else None)
    else:
            plt.plot(x_data, y_datas, marker='o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if legends:
        plt.legend()
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.savefig('Derived-results/' + title.replace(" ", "-") + '.png')
    plt.close()


if __name__ == '__main__':
    # Batch size results
    batch_size = [256, 512, 1024, 2048]
    acc_batch_size = [88.72, 88.22,87.7, 86.58]
    ce_acc_batch_size = [88.95, 87.89, 87.73, 85.26]
    create_plot(batch_size, [acc_batch_size, ce_acc_batch_size], "Batch Size", "Accuracy", "Batch size impact on Test Accuracy", legends=["SupCon", "Cross-Entropy"], ylim=[84,90]
                )
    # create_plot(batch_size, ce_acc_batch_size, "Batch Size", "Accuracy", "Batch size impact on test accuracy")

    # Tau/temp results
    tau = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16]
    acc_tau = [89.72, 89.75, 89.95, 90.04, 89.54, 89.65, 89.9, 89.77]
    create_plot(tau, acc_tau, "τ (Temperature)", "Accuracy", "τ impact on Test Accuracy", ylim=[89.2,90.5]
                )

    # Epoch results
    epoch = [50, 100, 150, 200, 250, 300, 350]
    acc_epoch = [87.33, 88.99, 89.18, 89.77, 89.9, 90.28, 90.29]
    ce_acc_epoch = [85.39, 87.76, 89.66, 89.13, 89.53, 89.67, 90.06]
    create_plot(epoch, [acc_epoch, ce_acc_epoch], "Epoch", "Accuracy", "Test Accuracy per Epoch", legends=["SupCon", "Cross-Entropy"], ylim=[84.5,91.2]
                )