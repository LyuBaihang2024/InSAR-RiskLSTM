import unittest
import torch
from src.models.spatial_attention import SpatialAttention, MultiScaleAttention
from src.models.lstm_predictor import LSTMTemporalModel, BidirectionalLSTM
from src.models.feature_fusion import FeatureFusion

class TestModels(unittest.TestCase):
    def setUp(self):
        self.input_dim = 128
        self.output_dim = 64
        self.hidden_dim = 128
        self.batch_size = 16
        self.sequence_length = 10

    def test_spatial_attention(self):
        model = SpatialAttention(self.input_dim, self.output_dim)
        inputs = torch.rand(self.batch_size, self.input_dim)
        outputs = model(inputs)
        self.assertEqual(outputs.shape, (self.batch_size, self.input_dim))

    def test_multiscale_attention(self):
        model = MultiScaleAttention(self.input_dim, self.output_dim, num_scales=3)
        inputs = torch.rand(self.batch_size, self.input_dim)
        outputs = model(inputs)
        self.assertEqual(outputs.shape, (self.batch_size, self.input_dim))

    def test_lstm_temporal_model(self):
        model = LSTMTemporalModel(self.input_dim, self.hidden_dim, self.output_dim)
        inputs = torch.rand(self.batch_size, self.sequence_length, self.input_dim)
        outputs = model(inputs)
        self.assertEqual(outputs.shape, (self.batch_size, self.output_dim))

    def test_bidirectional_lstm(self):
        model = BidirectionalLSTM(self.input_dim, self.hidden_dim, self.output_dim)
        inputs = torch.rand(self.batch_size, self.sequence_length, self.input_dim)
        outputs = model(inputs)
        self.assertEqual(outputs.shape, (self.batch_size, self.output_dim))

    def test_feature_fusion(self):
        model = FeatureFusion(self.input_dim, self.input_dim, 64, self.output_dim)
        spatial_features = torch.rand(self.batch_size, self.input_dim)
        temporal_features = torch.rand(self.batch_size, self.input_dim)
        outputs = model(spatial_features, temporal_features)
        self.assertEqual(outputs.shape, (self.batch_size, self.output_dim))

if __name__ == '__main__':
    unittest.main()
