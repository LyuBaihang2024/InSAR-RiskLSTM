import torch
from torch.optim import Adam
from src.data.data_loader import get_dataloader
from src.models.spatial_attention import SpatialAttention
from src.models.lstm_predictor import LSTMTemporalModel
from src.models.feature_fusion import FeatureFusion

def train_model(dataloader, model, optimizer, criterion, epochs):
    for epoch in range(epochs):
        for insar_data, labels in dataloader:
            outputs = model(insar_data)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Example use case
if __name__ == "__main__":
    # Placeholder data
    dataloader = get_dataloader(torch.rand(100, 10), torch.rand(100, 1), batch_size=16)
    model = FeatureFusion(10, 10, 5)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    train_model(dataloader, model, optimizer, criterion, epochs=5)
1