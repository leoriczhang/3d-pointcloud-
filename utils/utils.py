import os
import numpy as np
import sklearn.metrics as metrics
import torch
import pickle as pkl

def plot_confusion_matrix(correct_labels, predicted_labels, mode, epoch, numClasses, path):

    # Concats
    predicted_labels = np.concatenate(predicted_labels)
    correct_labels = np.concatenate(correct_labels)
    
    # Confusion matrix:
    conf_mat = metrics.confusion_matrix(correct_labels, predicted_labels, [i for i in range(numClasses)])

    with open(os.path.join(path, "{}_{}.pkl".format(epoch, mode)), "wb") as f:
        pkl.dump(conf_mat, f)

    print(conf_mat)



def calculate_results(correct_labels, predicted_labels, loss_sum, results, mode, epoch):
    """
    Calculates normal, average and loss result based upon correct and predicted labels. Moreover, this information is saved
    unto the dictionaries containing results.
    """
    # Concats
    predicted_labels = np.concatenate(predicted_labels)
    correct_labels = np.concatenate(correct_labels)

    # Computes accuracy and loss based on metrics
    normal_accuracy = metrics.accuracy_score(correct_labels, predicted_labels)
    average_accuracy = metrics.balanced_accuracy_score(correct_labels, predicted_labels)


    # Save accuracy and loss to lists
    results.get('normal').append(normal_accuracy)
    results.get('average').append(average_accuracy)
    results.get('loss').append(loss_sum)

    print(f'Loss: {loss_sum}| {mode} acc: {normal_accuracy}| {mode} avg acc: {average_accuracy}')
    return normal_accuracy, average_accuracy


def save_at_highest_validation(normal_accuracy, best_valid_accuracy, epoch, current_path, model, filename="model_acc.t7"):
    """
    Checks if a new validation accuracy is better than the current best validation accuracy.
    If true, it saves the current model
    """
    if normal_accuracy > best_valid_accuracy:
        best_valid_accuracy = normal_accuracy
        print('[INFO] Saving Model...')
        np.savetxt(f'{current_path}/Epoch.txt', [epoch])
        torch.save(model, f'{current_path}/{filename}')
    return best_valid_accuracy


def save_at_lowest_loss(loss, best_valid_loss, epoch, current_path, model, filename="model_loss.t7"):
    """
    Checks if a new validation loss is better than the current best validation loss.
    If true, it saves the current model
    """
    if loss < best_valid_loss:
        best_valid_loss = loss
        print('[INFO] Saving Model...')
        np.savetxt(f'{current_path}/Epoch.txt', [epoch])
        torch.save(model, f'{current_path}/{filename}')
    return best_valid_loss


def save_to_csvs(current_path, mode, results):
    """
    Saves results unto CSVs
    """
    np.savetxt(f'{current_path}/{mode}_Accuracy_normal.csv', np.array(results.get('normal')))
    np.savetxt(f'{current_path}/{mode}_Accuracy_average.csv', np.array(results.get('average')))
    np.savetxt(f'{current_path}/{mode}_Loss.csv', np.array(results.get('loss')))


def create_folder_and_get_path(base_path = "./runs", exp_name = ""):
    """
    Creates a new folder based on the number of folders found. Moreover, it returns the path the newly created folder.
    """
    current_path = os.path.join(base_path, exp_name)

    if os.path.isdir(current_path):
        print("Experiment folder already exists.")
        return False, ""
    else:
        os.makedirs(current_path)

    return True, current_path