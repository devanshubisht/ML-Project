from matplotlib import pyplot as plt

def show_images(images, n_row=5, n_col=5, figsize=[12,12]):
    _, axs = plt.subplots(n_row, n_col, figsize=figsize)
    axs = axs.flatten()
    for img, ax in zip(images, axs):
        ax.imshow(img, cmap='gray')
    plt.show()
    
def dict_train_test_split(X_dict, y, ratio=0.9):
    N_train = int(len(y)*ratio)
    
    X_dict_train = {k:v[:N_train] for k,v in X_dict.items()}
    y_train = y[:N_train]

    X_dict_test = {k:v[N_train:] for k,v in X_dict.items()}
    y_test = y[N_train:]
    
    return X_dict_train, y_train, X_dict_test, y_test
