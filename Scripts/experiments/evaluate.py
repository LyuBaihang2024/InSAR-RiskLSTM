import torch

def evaluate_model(dataloader, model):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for insar_data, labels in dataloader:
            outputs = model(insar_data)
            total_loss += ((outputs - labels) ** 2).mean().item()
    print(f"Evaluation Loss: {total_loss / len(dataloader)}")
1