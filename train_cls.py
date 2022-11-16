import argparse
import os
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from models.dgcnn import DGCNN
from models.pointnet import PointNet, feature_transform_regularizer
import transforms as transforms
from data import PointCloudDataset
from utils import utils



def train(mode, criterion, optimizer, scheduler, dataloader, loglist, numClasses, epoch, epochs, current_path, model, device, model_type, transform_regularization):
    print(f'\n[INFO] Epoch: {epoch+1}/{epochs} | {mode}...')

    model.train()

    loss_sum = 0.0
    pointclouds_processed = 0
    predicted_labels = []
    correct_labels = []

    # Goes through all data
    for data, label in tqdm(dataloader):
        data = data.to(device)
        data = data.float()
        label = label.to(device).squeeze()

        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]

        optimizer.zero_grad()


        if model_type == "pointnet":
            output,  trans, trans_feat = model(data)
            loss = criterion(output, label)

            if transform_regularization > 0.0:
                loss += feature_transform_regularizer(trans_feat)*transform_regularization
        else:
            output = model(data)
            loss = criterion(output, label)
        
        loss.backward()
        optimizer.step()

        predictions = output.max(dim=1)[1]
        pointclouds_processed += batch_size

        loss_sum += loss.item() * batch_size
        correct_labels.append(label.cpu().numpy())
        predicted_labels.append(predictions.detach().cpu().numpy())

    loss_sum = loss_sum * 1.0 / pointclouds_processed
    
    accuracy, balanced_accuracy = utils.calculate_results(correct_labels, predicted_labels, loss_sum, loglist, mode, epoch)
    utils.plot_confusion_matrix(correct_labels, predicted_labels, mode, epoch, numClasses, current_path)

    return None, None


def validation(mode, data_mode, criterion, dataloader, loglist, best_valid_accuracy, best_valid_balanced_accuracy, best_valid_loss, numClasses, epoch, epochs, current_path, model, device, model_type, transform_regularization):
    print(f'\n[INFO] Epoch: {epoch+1}/{epochs} | {mode}...')

    model.eval()

    loss_sum = 0.0
    pointclouds_processed = 0
    predicted_labels = []
    correct_labels = []

    # Goes through all data
    with torch.no_grad():
        for data, label in tqdm(dataloader):

            data = data.to(device)
            data = data.float()
            label = label.to(device).squeeze()

            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]


            if model_type == "pointnet":
                output,  trans, trans_feat = model(data)
                loss = criterion(output, label)

                if transform_regularization > 0.0:
                    loss += feature_transform_regularizer(trans_feat)*transform_regularization
            else:
                output = model(data)
                loss = criterion(output, label)


            predictions = output.max(dim=1)[1]
            pointclouds_processed += batch_size

            loss_sum += loss.item() * batch_size
            correct_labels.append(label.cpu().numpy())
            predicted_labels.append(predictions.detach().cpu().numpy())

    loss_sum = loss_sum * 1.0 / pointclouds_processed
    accuracy, balanced_accuracy = utils.calculate_results(correct_labels, predicted_labels, loss_sum, loglist, mode, epoch)
    utils.plot_confusion_matrix(correct_labels, predicted_labels, mode, epoch, numClasses, current_path)

    best_valid_accuracy = utils.save_at_highest_validation(accuracy, best_valid_accuracy, epoch, current_path, model, filename="model_{}_acc.t7".format(data_mode))
    best_valid_balanced_accuracy = utils.save_at_highest_validation(balanced_accuracy, best_valid_balanced_accuracy, epoch, current_path, model, filename="model_{}_balanced_acc.t7".format(data_mode))
    best_valid_loss = utils.save_at_lowest_loss(loss_sum, best_valid_loss, epoch, current_path, model, filename="model_{}_loss.t7".format(data_mode))
    return best_valid_accuracy, best_valid_balanced_accuracy, best_valid_loss




def main(args):

    isBinary = False
    if args.training_mode == "binary":
        args.num_classes = 2
        isBinary = True

    if args.exp_name == "":
        args.exp_name = "{}_{}_{:.6f}_{:.6f}".format(args.model, args.num_classes, args.lr, args.wd)

    status, current_path = utils.create_folder_and_get_path(exp_name=args.exp_name)

    if not status:
        return 0

    with open(os.path.join(current_path, "settings.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        print(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        print('Using CPU')
                           
    device = torch.device("cuda" if args.cuda else "cpu")


    # Initialize model
    if args.model == "pointnet":
        model = PointNet(args.num_classes, emb_dims=args.emb_dims, dropout_rate=args.dropout, feature_transform=args.transform_regularization > 0.0)
    elif args.model =="dgcnn":
        model = DGCNN(args.num_classes, emb_dims=args.emb_dims, dropout_rate=args.dropout, k=args.k)
    else:
        raise Exception("Not implemented")

    # Load model weights if provided
    if args.model_path != "":
        print("Loading model: {}".format(os.path.join(args.model_path, "model_Both_loss.t7")))
        model = torch.load(os.path.join(args.model_path, "model_Both_loss.t7"))
    
    print(model)
    model = model.to(device)

    # Setup optimizer
    if args.use_sgd:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Data pipeline

    ## Compose data augmentations and transforms
    train_transforms  = T.Compose([transforms.Normalize(), transforms.RandomNoise()])
    valid_transforms  = T.Compose([transforms.Normalize()])

    ## Initalize datasets
    train_dataset = PointCloudDataset(dataDir = args.data_path, partition='Training', num_points=1024, transforms = train_transforms, data_type = args.training_data, binary_data = isBinary, data_ratio=args.data_ratio)
    valid_real_dataset = PointCloudDataset(dataDir = args.data_path, partition='Validation', num_points=1024, transforms = valid_transforms, data_type = ["real"], binary_data = isBinary)
    valid_synthetic_dataset = PointCloudDataset(dataDir = args.data_path, partition='Validation', num_points=1024, transforms = valid_transforms, data_type = ["synthetic"], binary_data = isBinary)
    valid_dataset = PointCloudDataset(dataDir = args.data_path, partition='Validation', num_points=1024, transforms = valid_transforms, data_type = ["synthetic","real"], binary_data = isBinary)

    ## Adjust batch size if necessary
    if len(train_dataset) < args.batch_size:
        args.batch_size = len(train_dataset)

    ## Set up dataloaders
    train_loader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=True)
    validation_real_loader = DataLoader(valid_real_dataset, num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=False)
    validation_synthetic_loader = DataLoader(valid_synthetic_dataset, num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=False)
    validation_loader = DataLoader(valid_dataset, num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=False)


    # LR Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr/100)

    # Loss Function
    weight_tensor = torch.from_numpy(train_dataset.class_weights).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)


    # Dicts to log information
    best_valid_real_accuracy = 0.0
    best_valid_real_balanced_accuracy = 0.0
    best_valid_real_loss = np.inf
    train_log = {'normal': [], 'average': [], 'loss': []}

    best_valid_real_accuracy = 0.0
    best_valid_real_balanced_accuracy = 0.0
    best_valid_real_loss = np.inf
    validation_real_log = {'normal': [], 'average': [], 'loss': []}


    best_valid_synthetic_accuracy = 0.0
    best_valid_synthetic_balanced_accuracy = 0.0
    best_valid_synthetic_loss = np.inf
    validation_synthetic_log = {'normal': [], 'average': [], 'loss': []}


    best_valid_accuracy = 0.0
    best_valid_balanced_accuracy = 0.0
    best_valid_loss = np.inf
    validation_log = {'normal': [], 'average': [], 'loss': []}


    # Run Epochs
    for epoch in range(args.epochs):
        # Training
        train('Training', criterion, optimizer, scheduler, train_loader, train_log, args.num_classes, epoch, args.epochs, current_path, model, device, args.model, args.transform_regularization)
        scheduler.step()
        print("Learning Rate {}".format(scheduler.get_lr()))
        # Validation
        best_valid_synthetic_accuracy, best_valid_synthetic_balanced_accuracy, best_valid_synthetic_loss = validation('Validation-Synthetic', "Synthetic", criterion, validation_synthetic_loader, validation_synthetic_log, best_valid_synthetic_accuracy, best_valid_synthetic_balanced_accuracy, best_valid_synthetic_loss, args.num_classes, epoch, args.epochs, current_path, model, device, args.model, args.transform_regularization)
        best_valid_real_accuracy, best_valid_real_balanced_accuracy, best_valid_real_loss = validation('Validation-Real', "Real", criterion, validation_real_loader, validation_real_log, best_valid_real_accuracy, best_valid_real_balanced_accuracy, best_valid_real_loss, args.num_classes, epoch, args.epochs, current_path, model, device, args.model, args.transform_regularization)
        best_valid_accuracy, best_valid_balanced_accuracy, best_valid_loss = validation('Validation', "Both", criterion, validation_loader, validation_log, best_valid_accuracy, best_valid_balanced_accuracy, best_valid_loss, args.num_classes, epoch, args.epochs, current_path, model, device, args.model, args.transform_regularization)

        # Log information into CSVs
        utils.save_to_csvs(current_path, 'Train', train_log)
        utils.save_to_csvs(current_path, 'Validation-Synthetic', validation_synthetic_log)
        utils.save_to_csvs(current_path, 'Validation-Real', validation_real_log)
        utils.save_to_csvs(current_path, 'Validation', validation_log)

    print("Validation: Normal Acc: {}\t Balanced Acc: {}\t Loss: {}".format(best_valid_accuracy, best_valid_balanced_accuracy, best_valid_loss))
    print("Validation-Synthetic: Normal Acc: {}\t Balanced Acc: {}\t Loss: {}".format(best_valid_synthetic_accuracy, best_valid_synthetic_balanced_accuracy, best_valid_synthetic_loss))
    print("Validation-Real: Normal Acc: {}\t Balanced Acc: {}\t Loss: {}".format(best_valid_real_accuracy, best_valid_real_balanced_accuracy, best_valid_real_loss))



if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Setting for Point Cloud Classification')
    parser.add_argument('--exp_name', type=str, default='', metavar='N',
                        help='Name of the experiment')

    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]. (Default: dgcnn)')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--data_path', type=str, default='./data', metavar='N',
                        help='Dataset path')
    parser.add_argument('--num_classes', type=int, default=4, metavar='N',
                        help='Number of classes')
    parser.add_argument('--training_data', type=str, default=["real"], metavar='N',
                        help='Training data type')
    parser.add_argument('--training_mode', type=str, default="multi-class", metavar='N',
                        help='Training mode')
    parser.add_argument('--data_ratio', type=float, default=1.0, metavar='N',
                        help='Ratio of training data to use')

    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch. (Default: 32)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='Number of episode to train. (Default: 50)')

    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='Learning rate. (Default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum. (Default: 0.9)')
    parser.add_argument('--wd', type=float, default=0.001, metavar='LR',
                        help='Weight Decay. (Default: 0.001)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate. (Default: 0.5)')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings. (Default: 1024)')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN. (Default: 20)')
    parser.add_argument('--transform_regularization', type=float, default=0.001, metavar='N',
                        help='Regularization of the fature transform in PointNet. (Default: 0.001)')

    args = parser.parse_args()

    main(args)