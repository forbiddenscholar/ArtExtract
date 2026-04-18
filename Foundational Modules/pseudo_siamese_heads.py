import torch
import torch.nn as nn

class PseudoSiamesePatchEmbed(nn.Module):
    def __init__(self, patch_size = 4, embed_dim = 96):
        super().__init__()
        
        # for the 3-channel input tensor
        self.embed_rgb = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # for the 8-channel input tensor
        self.embed_msi = nn.Conv2d(
            in_channels=8,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, rgb_tensor, msi_tensor):
        rgb_features = self.embed_rgb(rgb_tensor)
        msi_features = self.embed_msi(msi_tensor)
        
        # we are skipping the flatten(2) and transpose(1,2)  part here
        # because the swin block is not integrated yet, 
        # here we are only making heads to project the different modalities into a same space
        return rgb_features, msi_features        
    
if __name__== "__main__":
    model = PseudoSiamesePatchEmbed(patch_size=4, embed_dim=96)
    
    dummy_rgb = torch.randn(1, 3, 256, 256) # (B, C_in, H, W)
    dummy_msi = torch.randn(1, 8, 256, 256) # (B, C_in, H, W)
    
    out_rgb, out_msi = model(dummy_rgb, dummy_msi)
    
    print(f"Input RGB shape: {dummy_rgb.shape} --> Embedded RGB shape: {out_rgb.shape}")
    print(f"Input MSI shape: {dummy_msi.shape} --> Embedded MSI shape: {out_msi.shape}")