import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class ViTEncoder(nn.Module):
    """Vision Transformer encoder for camera images"""
    def __init__(self, pretrained=True, patch_size=16, hidden_dim=768, output_dim=256):
        super(ViTEncoder, self).__init__()
        
        # Load pre-trained ViT model from torchvision
        # We'll use the base variant but smaller models can be used for speed
        self.vit = models.vit_b_16(pretrained=pretrained)
        
        # Adjust patch size if needed
        self.patch_size = patch_size
        
        # Remove the classification head
        self.vit.heads = nn.Identity()
        
        # Add a projection layer to get desired output dimension
        self.projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Forward pass for the ViT encoder
        
        Args:
            x (torch.Tensor): Image tensor of shape [B, 3, H, W]
            
        Returns:
            torch.Tensor: Image features of shape [B, output_dim]
        """
        # Extract features with ViT
        features = self.vit(x)  # [B, hidden_dim]
        
        # Project to desired output dimension
        features = self.projection(features)  # [B, output_dim]
        
        return features


class PointNetEncoder(nn.Module):
    """PointNet-based encoder for LiDAR point clouds"""
    def __init__(self, input_channels=3, output_dim=256):
        super(PointNetEncoder, self).__init__()
        
        # Input channels: at minimum x,y,z (3), could include intensity (4)
        self.input_channels = input_channels
        
        # MLP for point-wise feature extraction
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # MLP for global feature extraction
        self.mlp2 = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        
        # Projection to output dimension
        self.projection = nn.Linear(1024, output_dim)
        
    def forward(self, x):
        """
        Forward pass for the PointNet encoder
        
        Args:
            x (torch.Tensor): Point cloud tensor of shape [B, N, C] 
                              where N is the number of points and C is input_channels
                              
        Returns:
            torch.Tensor: Point cloud features of shape [B, output_dim]
        """
        # Ensure input is in the right format
        if x.dim() == 3:
            # [B, N, C] -> [B, C, N]
            x = x.transpose(1, 2)
        
        # Point-wise feature extraction
        point_features = self.mlp1(x)  # [B, 256, N]
        
        # Global feature extraction with max pooling
        global_features = self.mlp2(point_features)  # [B, 1024, N]
        global_features = torch.max(global_features, dim=2, keepdim=False)[0]  # [B, 1024]
        
        # Project to output dimension
        output = self.projection(global_features)  # [B, output_dim]
        
        return output


class SensorFusionModule(nn.Module):
    """Module for fusing camera and LiDAR features"""
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=512):
        super(SensorFusionModule, self).__init__()
        
        # Self-attention for cross-modal feature fusion
        self.self_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8)
        
        # Feature transformation layers
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, camera_features, lidar_features):
        """
        Forward pass for the fusion module
        
        Args:
            camera_features (torch.Tensor): Camera features of shape [B, input_dim]
            lidar_features (torch.Tensor): LiDAR features of shape [B, input_dim]
            
        Returns:
            torch.Tensor: Fused features of shape [B, output_dim]
        """
        batch_size = camera_features.size(0)
        
        # Stack features for self-attention
        # We need to reshape to [sequence_length, batch_size, input_dim]
        # Here sequence_length = 2 (camera + lidar)
        stacked_features = torch.stack([camera_features, lidar_features], dim=0)  # [2, B, input_dim]
        
        # Apply self-attention
        attn_output, _ = self.self_attn(stacked_features, stacked_features, stacked_features)  # [2, B, input_dim]
        
        # Reshape back to [B, 2*input_dim]
        attn_output = attn_output.transpose(0, 1).reshape(batch_size, -1)  # [B, 2*input_dim]
        
        # Apply fusion MLP
        fused_features = self.fusion_mlp(attn_output)  # [B, output_dim]
        
        return fused_features


class MultiModalPerceptionModel(nn.Module):
    """
    Multi-Modal Perception Model for autonomous driving
    Fuses camera images and LiDAR point clouds
    """
    def __init__(self, 
                 camera_feat_dim=256, 
                 lidar_feat_dim=256, 
                 fusion_output_dim=512,
                 num_classes=10,
                 lidar_input_channels=4,
                 use_calibration=True):
        super(MultiModalPerceptionModel, self).__init__()
        
        # Vision Transformer for camera images
        self.camera_encoder = ViTEncoder(
            pretrained=True, 
            patch_size=16, 
            hidden_dim=768, 
            output_dim=camera_feat_dim
        )
        
        # PointNet for LiDAR point clouds
        self.lidar_encoder = PointNetEncoder(
            input_channels=lidar_input_channels, 
            output_dim=lidar_feat_dim
        )
        
        # Sensor fusion module
        self.fusion_module = SensorFusionModule(
            input_dim=camera_feat_dim,  # Assuming same dimension for both
            hidden_dim=fusion_output_dim,
            output_dim=fusion_output_dim
        )
        
        # Task heads
        # In this example, we'll implement two common perception tasks:
        # 1. Object detection (bounding boxes)
        # 2. Road segmentation
        
        # Object detection head
        self.detection_head = nn.Sequential(
            nn.Linear(fusion_output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num