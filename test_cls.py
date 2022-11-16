import os
import argparse
import utils
import json
import numpy as np
from sklearn import metrics
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from models.dgcnn import DGCNN
from models.pointnet import PointNet, feature_transform_regularizer
import transforms as transforms
from data import PointCloudDataset

def test(args):
    current_path = os.path.dirname(args.model_path)

    with open(os.path.join(current_path, "settings.txt"), 'r') as f:
        settings = json.load(f)

    args.exp_name = settings["exp_name"]


    args.cuda = torch.cuda.is_available()
    if args.cuda:
        print(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(settings["seed"])
    else:
        print('Using CPU')
                           
    device = torch.device("cuda" if args.cuda else "cpu")


    numClass = settings["num_classes"]
    modelType = settings["model"]

    if modelType == "pointnet":
        model = PointNet(numClass, emb_dims=settings["emb_dims"], dropout_rate=settings["dropout"], feature_transform=settings["transform_regularization"] > 0.0)
    elif modelType =="dgcnn":
        model = DGCNN(numClass, emb_dims=settings["emb_dims"], dropout_rate=settings["dropout"], k=settings["k"])
    else:
        raise Exception("Not implemented")

    model_state_path = os.path.join(current_path, "model_Both_loss.t7")

    # load model
    model = torch.load(model_state_path)
    model.to(device)
    model.eval()

 
    # DataLoaders
    test_transforms  = T.Compose([transforms.Normalize()])

    test_real_dataset = PointCloudDataset(dataDir = args.data_path, partition='Testing', num_points=1024, transforms = test_transforms, data_type = ["real"], binary_data = False)
    test_synthetic_dataset = PointCloudDataset(dataDir = args.data_path, partition='Testing', num_points=1024, transforms = test_transforms, data_type = ["synthetic"], binary_data = False)
    test_dataset = PointCloudDataset(dataDir = args.data_path, partition='Testing', num_points=1024, transforms = test_transforms, data_type = ["synthetic","real"], binary_data = False)

    test_real_loader = DataLoader(test_real_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_synthetic_loader = DataLoader(test_synthetic_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False, drop_last=False)

    data_settings = [("Real", test_real_loader), ("Synthetic", test_synthetic_loader), ("All", test_loader)]


    with torch.no_grad():
        for data_setting, data_loader in data_settings:
            predicted_labels = []
            correct_labels = []

            baseDir = os.path.join(current_path, data_setting)
            if not os.path.isdir(baseDir):
                os.makedirs(baseDir)

            incorrectPredDir = os.path.join(baseDir, "incorrect_predictions")
            if not os.path.isdir(incorrectPredDir):
                os.makedirs(incorrectPredDir)

            for data, label in tqdm(data_loader):
                data = data.to(device)
                data = data.float()
                labels = label.to(device).squeeze()

                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]

                if modelType == "pointnet":
                    output, _, _ = model(data)
                else:
                    output = model(data)

                preds = output.max(dim=1)[1]
                correct_labels.append(labels.cpu().numpy())
                predicted_labels.append(preds.detach().cpu().numpy())

                for i in range(len(labels)):
                    if labels[i] != preds[i]:
                        temp_data = data[i].permute(1, 0)
        
                        num_csvs = len(os.listdir(incorrectPredDir))
                        np.savetxt(os.path.join(incorrectPredDir, "prediction{}_{}.csv".format(labels[i], num_csvs)), temp_data.cpu().numpy())

            correct_labels = np.concatenate(correct_labels)
            predicted_labels = np.concatenate(predicted_labels)
            cm = metrics.confusion_matrix(correct_labels, predicted_labels)
            print(cm)

            precision, recall, f1, support = metrics.precision_recall_fscore_support(correct_labels, predicted_labels, average="weighted")
            normal_accuracy = metrics.accuracy_score(correct_labels, predicted_labels)
            average_accuracy = metrics.balanced_accuracy_score(correct_labels, predicted_labels)

            np.savetxt(os.path.join(baseDir, "ConfusionMatrix.txt"), cm)

            with open(os.path.join(baseDir, "metrics.txt"), "w") as f:
                f.write("Precision: {}\n".format(precision))
                f.write("Recall: {}\n".format(recall))
                f.write("F1: {}\n".format(f1))
                f.write("Support: {}\n".format(support))
                f.write("Average: {}\n".format(normal_accuracy))
                f.write("Balanced Average: {}\n".format(average_accuracy))
            
            print(precision, recall, f1, support, normal_accuracy, average_accuracy)



if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Settings for Point Cloud Classification Test')
    parser.add_argument('--exp_name', type=str, default='', metavar='N',
                        help='Name of the experiment')

    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--data_path', type=str, default='', metavar='N',
                        help='Dataset path')

    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch. (Default: 32)')

    args = parser.parse_args()

    test(args)