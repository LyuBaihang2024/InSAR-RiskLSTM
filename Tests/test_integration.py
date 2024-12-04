import unittest
import torch
from src.data.data_loader import get_dataloader
from src.models.spatial_attention import SpatialAttention
from src.models.lstm_predictor import LSTMTemporalModel
from src.models.feature_fusion import FeatureFusion


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.batch_size = 16
        self.input_dim = 128
        self.hidden_dim = 64
        self.output_dim = 1
        self.sequence_length = 10
        self.data = torch.rand(100, self.sequence_length, self.input_dim)
        self.labels = torch.randint(0, 2, (100, 1))

    def test_full_pipeline(self):
        # Create DataLoader
        dataloader = get_dataloader(self.data, self.labels, self.batch_size)

        # Initialize models
        spatial_attention = SpatialAttention(self.input_dim, self.hidden_dim)
        lstm_model = LSTMTemporalModel(self.hidden_dim, self.hidden_dim, self.hidden_dim)
        fusion_model = FeatureFusion(self.hidden_dim, self.hidden_dim, 64, self.output_dim)

        for data_batch, labels_batch in dataloader:
            spatial_features = spatial_attention(data_batch[:, -1, :])
            temporal_features = lstm_model(data_batch)
            output = fusion_model(spatial_features, temporal_features)

            # Check output dimensions
            self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_training_step(self):
        from torch.optim import Adam
        criterion = torch.nn.BCELoss()
        spatial_attention = SpatialAttention(self.input_dim, self.hidden_dim)
        lstm_model = LSTMTemporalModel(self.hidden_dim, self.hidden_dim, self.hidden_dim)
        fusion_model = FeatureFusion(self.hidden_dim, self.hidden_dim, 64, self.output_dim)
        optimizer = Adam(list(spatial_attention.parameters()) +
                         list(lstm_model.parameters()) +
                         list(fusion_model.parameters()), lr=0.001)

        dataloader = get_dataloader(self.data, self.labels, self.batch_size)
        for data_batch, labels_batch in dataloader:
            spatial_features = spatial_attention(data_batch[:, -1, :])
            temporal_features = lstm_model(data_batch)
            predictions = fusion_model(spatial_features, temporal_features)

            loss = criterion(predictions, labels_batch.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.assertGreater(loss.item(), 0)


if __name__ == '__main__':
    unittest.main()
