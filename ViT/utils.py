import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # Guaranteed results are reproducible
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # Traverse folders, each folder corresponds to a category
    land_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # Sort to ensure consistent order
    land_class.sort()
    # Generate category names and corresponding numerical indexes
    class_indices = dict((k, v) for v, k in enumerate(land_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # Store all image paths of the training set
    train_images_label = []  # Store the index information corresponding to the training set images
    val_images_path = []  # Store all image paths of the validation set
    val_images_label = []  # Store the index information corresponding to the validation set images
    every_class_num = []  # Store the total number of samples for each category
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # Image file suffix type
    # Iterate through the files in each folder
    for cla in land_class:
        cla_path = os.path.join(root, cla)
        # Traverse to obtain all supported file paths
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # Sort to ensure consistent order
        images.sort()
        # Get the index corresponding to this category
        image_class = class_indices[cla]
        # Record the number of samples in this category
        every_class_num.append(len(images))
        # Proportional random sampling of validation samples
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            # If the path is in the sampled validation set samples, then store it in the validation set
            if img_path in val_path:  
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # Otherwise, store it in the training set
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # Draw a bar chart of the number of each category
        plt.bar(range(len(land_class)), every_class_num, align='center')
        # Use category names to represent the x-axis
        plt.xticks(range(len(land_class)), land_class)
        # Add numerical labels to bar chart
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # Set x coordinate
        plt.xlabel('image class')
        # Set y coordinate
        plt.ylabel('number of images')
        # Set the title of the bar chart
        plt.title('land class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # # Conversion of tensor dimensions: [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # Inverse normalization
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # Remove the scale from the x-axis
            plt.yticks([])  # Remove the scale from the y-axis
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # Accumulated loss
    accu_num = torch.zeros(1).to(device)   # Accumulated number of correct predictions
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # Accumulated number of correct predictions
    accu_loss = torch.zeros(1).to(device)  # Accumulated loss

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
