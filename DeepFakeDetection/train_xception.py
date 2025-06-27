import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import torch.optim as optim
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as T
import random
from PIL import Image
import io
import os
from torch.utils.data import Dataset
import numpy as np
import pretrainedmodels
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import kornia.augmentation as K
import torchvision.transforms.functional as TF
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import pandas as pd 
from torchmetrics.classification import BinaryEER
import time
from torchvision.transforms import ToPILImage, ToTensor
import timm

torch.backends.cudnn.benchmark = True

# Windows environ
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#### Constants
FACE_CROPS = "test/"
VALIDATION_SET = "validation_data/"
TEST_SET = "test_data/"
BATCH_SIZE = 32
UNFORZEN_LAYERS = 18
DECISION_TRESHOLD = 0.6

LEARNING_RATE = 3e-4
BETA1 = 0.9
BETA2 = 0.999
T_MAX = 10

eer = BinaryEER(thresholds=None)

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: raw logits (B,) or (B, 1)
        targets: binary labels (B,) or (B, 1)
        """
        if inputs.ndim != targets.ndim:
            targets = targets.view_as(inputs)

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

def jpeg_quality_augment_video(video_tensor: torch.Tensor, quality_range=(10, 50)):
    T, C, H, W = video_tensor.shape
    to_pil = ToPILImage()
    to_tensor = ToTensor()
    
    augmented = torch.empty_like(video_tensor)
    
    for t in range(T):
        frame = video_tensor[t]
        pil_img = to_pil(frame)
        quality = random.randint(*quality_range)
        
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        compressed = Image.open(buffer)
        augmented[t] = to_tensor(compressed)
    
    return augmented

def bootstrap_ci(y_true, y_pred, metric_func,num_bootstrap, alpha, random_seed):
    rng = np.random.default_rng(random_seed)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    scores = []
    for _ in range(num_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        sample_true = y_true[indices]
        sample_pred = y_pred[indices]
        score = metric_func(sample_true, sample_pred)
        scores.append(score)
    scores = np.array(scores)
    lower = np.percentile(scores, 100 * (alpha / 2))
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    mean = np.mean(scores)
    std = np.std(scores)
    return mean, lower, upper, std

def load_npz_tensor(path):
    data = np.load(path)['frames']
    if data.shape == (10, 224, 224, 3):
        data = np.transpose(data, (0, 3, 1, 2))  
    if data.shape == (10, 224, 3, 224):
        data = np.transpose(data, (0, 2, 1, 3)) 
    return torch.from_numpy(data)

# Class to include the Sigmoid
class XceptionBinaryClassifier(nn.Module):
    def __init__(self, model):
        super(XceptionBinaryClassifier, self).__init__()
        self.model = model      

    def forward(self, x):
        x = self.model(x)
        return x

#### Augmentation
# Augmentation used in training.
transform_training = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    T.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.0),
    T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    T.RandomErasing(p=0.7, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# No augmentation in testing/validation.
transform_val_test = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#### Dataset processing
class DFDCCropsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.data_dir = FACE_CROPS
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)
    
    def getOriginalImage(self, idx):
        return self.file_list[idx]

    def __getitem__(self, idx):
        file_path = self.file_list[idx]

        # Loading frames from each 4D tensor in the .npz files.
        frames = load_npz_tensor(file_path)

        if self.transform:
            #if random.random() < 0.4:
            #    frames = jpeg_quality_augment_video(frames)

            frames = torch.stack([
                self.transform(f) for f in frames
            ])

        filename = self.file_list[idx]
        label = 0 if "REAL" in filename else 1 

        return frames, label

#### Splitting the examples.
train_files = [ FACE_CROPS + f for f in os.listdir(FACE_CROPS) ]
val_files   = [ VALIDATION_SET + f for f in os.listdir(VALIDATION_SET) ]
test_files  = [ TEST_SET + f for f in os.listdir(TEST_SET) ]

train_dataset = DFDCCropsDataset(train_files, transform=transform_training)
val_dataset = DFDCCropsDataset(val_files, transform=transform_val_test)
test_dataset = DFDCCropsDataset(test_files, transform=transform_val_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=False)

# BCEWithLogitsLossWithLabelSmoothing
class BCEWithLogitsLossWithLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.05):
        super(BCEWithLogitsLossWithLabelSmoothing, self).__init__()
        self.smoothing = smoothing
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        loss = self.bce_loss(inputs, targets)
        return loss.mean()

def train_one_epoch(model, criterion, train_loader, epoch_index, device, tb_writer, optimizer, scheduler, scaler):
    model.train()
    running_loss = 0.
    last_loss = 0.

    all_preds = []
    all_probs = []
    all_labels = []

    for i, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch_index}"):
        data = data.to(device)
        target = target.to(device).float()

        optimizer.zero_grad()

        batch_loss = 0.

        # Loop over each frame for every video in the batch.
        # AMP on
        with autocast(device_type=str(device)): 
            loss = 0.
            for frame_index in range(data.size(1)):  
                frame = data[:, frame_index, :, :, :]
                outputs = model(frame).squeeze(1)
                loss += criterion(outputs, target)
            loss /= data.size(1)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_loss = loss.item()
        running_loss += batch_loss

        with torch.no_grad():
            logits = outputs.detach().cpu()
            probs = torch.sigmoid(logits).numpy()             
            preds = (probs > DECISION_TRESHOLD).astype(int)
            labels_np = target.detach().cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels_np)

        if i % BATCH_SIZE == 0:
            last_loss = running_loss / BATCH_SIZE
            print(f"batch {i + 1} loss: {last_loss:.4f}")
            tb_x = (epoch_index + 1) * len(train_loader)

            # Loss per Batch logged.
            tb_writer.add_scalar('Loss/Train', last_loss, tb_x)

            running_loss = 0.0

    # Computation of metrics at the end of the epoch.
    roc_auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    beer_score = eer(torch.tensor(all_probs), torch.tensor(all_labels).long())

    # Logging to Tensorboard.
    tb_writer.add_scalar('ROC_AUC/Train', roc_auc, tb_x)
    tb_writer.add_scalar('Accuracy/Train', acc, tb_x)
    tb_writer.add_scalar('F1/Train', f1, tb_x)
    tb_writer.add_scalar('EER/Train', beer_score, tb_x)

    #cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["REAL", "FAKE"])
    
    #fig, ax = plt.subplots(figsize=(6, 6))
    #disp.plot(ax=ax, cmap='Blues', colorbar=False)
    #plt.title(f"Confusion Matrix (Epoch {epoch_index})")

    # Log to TensorBoard
    #tb_writer.add_figure(f"Confusion Matrix (Epoch {epoch_index})", fig, global_step=epoch_index)

    scheduler.step()

    return roc_auc, last_loss, f1, beer_score

def validate_one_epoch(model, criterion, val_loader, epoch_index, device, tb_writer):
    model.eval()
    val_loss = 0.

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for i, (data, target) in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Validation Epoch {epoch_index}"):
            data = data.to(device)
            target = target.to(device).float()
            batch_loss = 0.
            logits_accum = []

            # Loop over each frame for every video in the batch.
            for frame_index in range(data.size(1)):  

                # Extracting the i-th frame for the entire batch.
                frame = data[:, frame_index, :, :, :] 
                outputs = model(frame).squeeze(1) 
                loss = criterion(outputs, target)  
                batch_loss += loss.item()
                logits_accum.append(outputs.detach())

            batch_loss /= data.size(1)
            val_loss += batch_loss
            
            logits = torch.stack(logits_accum, dim=0).mean(dim=0)
            probs = torch.sigmoid(logits.cpu()).numpy()
            preds = (probs > DECISION_TRESHOLD).astype(int)
            labels_np = target.detach().cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels_np)

    # Computation of metrics at the end of the epoch.
    avg_val_loss = val_loss / len(val_loader)
    roc_auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    beer_score = eer(torch.tensor(all_probs), torch.tensor(all_labels).long())

    # Logging to Tensorboard.
    tb_writer.add_scalar('Loss/Validation', avg_val_loss, epoch_index)
    tb_writer.add_scalar('ROC_AUC/Validation', roc_auc, epoch_index)
    tb_writer.add_scalar('Accuracy/Validation', acc, epoch_index)
    tb_writer.add_scalar('F1/Validation', f1, epoch_index)
    tb_writer.add_scalar('EER/Validation', beer_score, epoch_index)

    #cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["REAL", "FAKE"])
    
    #fig, ax = plt.subplots(figsize=(6, 6))
    #disp.plot(ax=ax, cmap='Oranges', colorbar=False)
    #plt.title(f"Validation Confusion Matrix (Epoch {epoch_index})")

    # Log to TensorBoard
    #tb_writer.add_figure(f"Validation Confusion Matrix (Epoch {epoch_index})", fig, global_step=epoch_index)

    return roc_auc, avg_val_loss, all_labels, all_preds, all_probs, f1, beer_score

def test(model, criterion, test_loader, device, tb_writer):
    model.eval()
    test_loss = 0.

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for i, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Testing"):
            data = data.to(device)
            target = target.to(device).float()
            batch_loss = 0.
            logits_accum = []

            # Loop over each frame for every video in the batch.
            for frame_index in range(data.size(1)):  

                # Extracting the i-th frame for the entire batch.
                frame = data[:, frame_index, :, :, :] 
                outputs = model(frame).squeeze(1) 
                loss = criterion(outputs, target)  
                batch_loss += loss.item()
                logits_accum.append(outputs.detach())

            batch_loss /= data.size(1)
            test_loss += batch_loss
            
            logits = torch.stack(logits_accum, dim=0).mean(dim=0)
            probs = torch.sigmoid(logits.cpu()).numpy()
            preds = (probs > DECISION_TRESHOLD).astype(int)
            labels_np = target.detach().cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels_np)

    # Computation of metrics at the end of the epoch.
    avg_test_loss = test_loss / len(test_loader)
    roc_auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    beer_score = eer(torch.tensor(all_probs), torch.tensor(all_labels).long())

    # Logging to Tensorboard.
    tb_writer.add_scalar('Loss/Testing', avg_test_loss)
    tb_writer.add_scalar('ROC_AUC/Testing', roc_auc)
    tb_writer.add_scalar('Accuracy/Testing', acc)
    tb_writer.add_scalar('F1/Testing', f1)
    tb_writer.add_scalar('EER/Testing', beer_score)

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["REAL", "FAKE"])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Oranges', colorbar=False)
    plt.title(f"Testing Confusion Matrix")

    # Log to TensorBoard
    tb_writer.add_figure(f"Testing Confusion Matrix", fig)

    return roc_auc, avg_test_loss, acc, all_labels, all_preds, all_probs, f1, beer_score

# Organize blocks and layers by unfreeze order
unfreeze_groups = [
    ["blocks.2"], 
    ["blocks.1"],
    ["blocks.0"],
    ["conv_stem", "bn1"],
]

def unfreeze_layer_groups(model, group_names):
    for name, param in model.named_parameters():
        if any(name.startswith(g) for g in group_names):
            param.requires_grad = True

if __name__ == "__main__":
    total_times = []
    timestamp = datetime.now().strftime('%d-%m_%H-%M-%S')
    writer = SummaryWriter('runs/deep_fake_{}'.format(timestamp))
    
    model_dir = f'models/{timestamp}'
    os.makedirs(model_dir, exist_ok=True)

    #criterion = BCEWithLogitsLossWithLabelSmoothing(smoothing=0.05)
    criterion = BinaryFocalLoss(alpha=0.25, gamma=2.0, reduction='mean')

    #### Model configurations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Only for the EfficientNet B3 experiment.
    model_en = timm.create_model('efficientnet_b3', pretrained=True)
    model_en.classifier = nn.Linear(model_en.classifier.in_features, 1)

    # Freezing everyting except head
    for name, param in model_en.named_parameters():
        if name.startswith("classifier") or name.startswith("blocks.3"):
            param.requires_grad = True
        else:
            param.requires_grad = False


    model = pretrainedmodels.__dict__['xception'](pretrained='imagenet')

    # Replacement of the last layer with an FC one.
    model.last_linear = nn.Linear(model.last_linear.in_features, 1)
    model = XceptionBinaryClassifier(model)

    #### Freezing the model's weights except for the last layer's ones.
    for param in model.model.parameters():
        param.requires_grad = False

    model.model.last_linear.requires_grad = True

    # Code to freeze all layers except the head classifier.
    for param in model.model.last_linear.parameters():
        param.requires_grad = True

    all_layers = list(model.model.modules())
    # Unfreeze parameters of the last 18 layers
    for layer in all_layers[-UNFORZEN_LAYERS:]:
        for param in layer.parameters(recurse=False):
            param.requires_grad = True

    # AMP
    scaler = GradScaler()

    EPOCHS = 10
    best_val_rocauc = 0.0
    best_epoch = 1

    efficent_exp = False

    model_used = None
    if efficent_exp:
        model_used = model_en
        print("Using EfficientNet B3")
    else:
        model_used = model
        print("Using XceptionNet")
    model_used.to(device)

    # Optimizer and Scheduler
    # , weight_decay = 1e-4
    optimizer = optim.AdamW(model_used.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    scheduler = CosineAnnealingLR(optimizer, T_max=T_MAX)

    print("## Beginning training for {0} EPOCHS.".format(EPOCHS))
    for epoch in range(1, EPOCHS+1):
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()

        if efficent_exp:
            if epoch == 4:
                unfreeze_layer_groups(model, ["blocks.2"])
            elif epoch == 5:
                unfreeze_layer_groups(model, ["blocks.1"])
            elif epoch == 6:
                unfreeze_layer_groups(model, ["blocks.0"])
            elif epoch == 7:
                unfreeze_layer_groups(model, ["conv_stem", "bn1"])

        epoch_train_rocauc, epoch_train_loss, epoch_train_f1, epoch_train_beer = train_one_epoch(model_used, criterion, train_loader, epoch, device, writer, optimizer, scheduler, scaler)
        epoch_val_rocauc, epoch_val_loss, all_labels_val, all_preds_val, all_probs_val, epoch_val_f1, epoch_val_beer = validate_one_epoch(model_used, criterion, val_loader, epoch, device, writer)

        print("----------End of Epoch----------")
        print("##########Raw Statistics########")
        print('Epoch {0} Loss:  Train - {1} Valid - {2}'.format(epoch, epoch_train_loss, epoch_val_loss))
        print('Epoch {0} AUROC: Train - {1} Valid - {2}'.format(epoch, epoch_train_rocauc, epoch_val_rocauc))
        print('Epoch {0} F1: Train - {1} Valid - {2}'.format(epoch, epoch_train_f1, epoch_val_f1))
        print('Epoch {0} EER: Train - {1} Valid - {2}'.format(epoch, epoch_train_beer, epoch_val_beer))
        
        print("##########CI Statistics#########")
        mean_auc, lower_auc, upper_auc, std_auc = bootstrap_ci(all_labels_val, all_probs_val, roc_auc_score, 1000, 0.05, 42)
        print(f"AUROC mean: {mean_auc:.3f}")
        print(f"AUROC std: {std_auc:.3f}")
        print(f"AUROC 95% CI: ({lower_auc:.3f}, {upper_auc:.3f})")
        print("---")

        mean_acc, lower_acc, upper_acc, std_acc = bootstrap_ci(all_labels_val, all_preds_val, accuracy_score, 1000, 0.05, 42)
        print(f"ACC mean: {mean_acc:.3f}")
        print(f"ACC std: {std_acc:.3f}")
        print(f"ACC 95% CI: ({lower_acc:.3f}, {upper_acc:.3f})")
        print("---")

        mean_f1, lower_f1, upper_f1, std_f1 = bootstrap_ci(all_labels_val, all_preds_val, f1_score, 1000, 0.05, 42)
        print(f"F1 mean: {mean_f1:.3f}")
        print(f"F1 std: {std_f1:.3f}")
        print(f"F1 95% CI: ({lower_f1:.3f}, {upper_f1:.3f})")

        print("--------------------------------")

        end_time = time.time()
        epoch_time_minutes = (end_time - start_time) / 60
        total_times.append(epoch_time_minutes)
        gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024**2  # in MB

        # Logging Memory and Time per Epoch
        writer.add_scalar('System/TrainTime_Minutes', epoch_time_minutes, epoch)
        writer.add_scalar('System/TrainTime_GPU_Memory_MB', gpu_memory_mb, epoch)

        print(f"[Epoch {epoch}] Train Time: {epoch_time_minutes:.2f} min | GPU Mem: {gpu_memory_mb:.2f} MB")

        if epoch_val_rocauc > best_val_rocauc:
            best_val_rocauc = epoch_val_rocauc
            best_epoch = epoch
            model_path = 'models/{0}/model_epoch_{1}.pt'.format(timestamp, epoch)
            torch.save(model_used.state_dict(), model_path)

    print("Logged as: {0}".format('runs/training/deep_fake_{}'.format(timestamp)))
    print(f"Training finished. Best model from epoch {best_epoch}.")
    print("## End of training.")

    print("-----------Testing-------------")
    start_time = time.time()
    # Loading the best model from training/validation.
    best_model_path = f'models/{timestamp}/model_epoch_{best_epoch}.pt'
    model_used.load_state_dict(torch.load(best_model_path, weights_only=True))
    model_used.to(device)

    # Run test evaluation
    test_rocauc, test_loss, test_acc, all_labels_test, all_preds_test, all_probs_test, test_f1, test_beer = test(model_used, criterion, test_loader, device, writer)

    end_time = time.time()
    test_time_minutes = (end_time - start_time) / 60
    total_times.append(test_time_minutes)
    gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024**2 

    print(f"[Epoch {epoch}] Test Time: {test_time_minutes:.2f} min | GPU Mem: {gpu_memory_mb:.2f} MB")

    # Logging Memory and Time per Epoch
    writer.add_scalar('System/TestTime_Minutes', test_time_minutes, epoch)
    writer.add_scalar('System/TestTime_GPU_Memory_MB', gpu_memory_mb, epoch)

    print("##########Raw Statistics########")
    print(f"Loss: {test_loss:.4f}")
    print(f"AUROC: {test_rocauc:.4f}")
    print(f"ACC: {test_acc:.4f}")
    print(f"F1: {test_f1:.4f}")
    print(f"EER: {test_beer:.4f}")

    print("##########CI Statistics#########")
    mean_auc, lower_auc, upper_auc, std_auc = bootstrap_ci(all_labels_test, all_probs_test, roc_auc_score, 1000, 0.05, 42)
    print(f"AUROC mean: {mean_auc:.3f}")
    print(f"AUROC std: {std_auc:.3f}")
    print(f"AUROC 95% CI: ({lower_auc:.3f}, {upper_auc:.3f})")
    print("---")

    mean_acc, lower_acc, upper_acc, std_acc = bootstrap_ci(all_labels_test, all_preds_test, accuracy_score, 1000, 0.05, 42)
    print(f"ACC mean: {mean_acc:.3f}")
    print(f"ACC std: {std_acc:.3f}")
    print(f"ACC 95% CI: ({lower_acc:.3f}, {upper_acc:.3f})")
    print("---")

    mean_f1, lower_f1, upper_f1, std_f1 = bootstrap_ci(all_labels_test, all_preds_test, f1_score, 1000, 0.05, 42)
    print(f"F1 mean: {mean_f1:.3f}")
    print(f"F1 std: {std_f1:.3f}")
    print(f"F1 95% CI: ({lower_f1:.3f}, {upper_f1:.3f})")

    print("--------------------------------")
    total_params = sum(p.numel() for p in model_used.parameters()) / 1e6
    print(f"Trainable Parameters: {total_params:.2f}M")
    print("## End of testing.")

    print("Debugging:")
    test_image_predictions = dict()
    for sample in range(len(all_labels_test)):
        video_name = test_dataset.getOriginalImage(sample)
        test_image_predictions[video_name] = {
            'video_name': video_name,
            'label': "REAL" if int(all_labels_test[sample]) == 0 else "FAKE",
            'predicted_label': "REAL" if int(all_preds_test[sample]) == 0 else "FAKE",
            'prob': float(all_probs_test[sample])
        }
    total_times.append(epoch_time_minutes)
    print(f"Total Time: {sum(total_times):.2f} min")

