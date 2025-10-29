import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn
from vinafood_dataset import VinafoodDataset, collate_fn
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import optim
from googlenet import GoogleNet
from torch.cuda.amp import autocast, GradScaler
from resnet18 import ResNet18
from pretrained_resnet import PretrainedResnet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(dataloader: DataLoader, model: nn.Module) -> dict:
    model.eval()
    predictions = []
    trues = []
    for item in dataloader:
        image: torch.Tensor = item['image'].to(device)
        label: torch.Tensor = item['label'].to(device)
        output: torch.Tensor = model(image)
        output = output.argmax(dim=-1)
        
        predictions.extend(output.cpu().numpy().tolist())
        trues.extend(label.cpu().numpy().tolist())
        
    return {
        'precision': precision_score(trues, predictions, average='macro', zero_division=0),
        'recall': recall_score(trues, predictions, average='macro', zero_division=0),
        'f1': f1_score(trues, predictions, average='macro', zero_division=0)
    }
    
if __name__ == '__main__':
    train_dataset = VinafoodDataset(
        image_path='./VinaFood21/train',
        image_size=(32, 32)
    )
    test_dataset = VinafoodDataset(
        image_path='./VinaFood21/test',
        image_size=(32,32)
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )
    print("Số lớp:", len(train_dataset.label2idx))
    print("Các lớp:", train_dataset.label2idx)

    num_classes = len(train_dataset.label2idx)
    #model = GoogleNet(num_classes=num_classes).to(device)
    #model = ResNet18(num_classes=num_classes).to(device)
    model = PretrainedResnet().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = GradScaler()

    num_epochs = 10
    best_score = 0.0 
    best_score_name = "f1"
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}:")

        losses = []
        model.train()
        for item in train_dataloader:
            image: torch.Tensor = item['image'].to(device)
            label: torch.Tensor = item['label'].to(device).long()
                        
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
              output: torch.Tensor = model(image)
              loss: torch.Tensor =  loss_fn(output, label)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            losses.append(loss.item())
        print(f"  Train loss: {np.mean(losses):.4f}")
        
        metrics = evaluate(test_dataloader, model)
        for metric_name in metrics:
            print(f"\t- {metric_name}: {metrics[metric_name]} ")
            
        if metrics[best_score_name] > best_score:
            best_score = metrics[best_score_name]
            torch.save(model.state_dict(), f'best_model.pth')
            print(f"  Best model saved with {best_score_name}: {best_score}")
            
        scheduler.step()
        