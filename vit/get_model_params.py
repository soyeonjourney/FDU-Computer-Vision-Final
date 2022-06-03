from models import ViT, ViT_Small

model = ViT(
    image_size=32,
    patch_size=4,
    num_classes=100,
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1,
)  # 12.718692

# model = ViT_Small(
#     image_size=32,
#     patch_size=4,
#     num_classes=100,
#     dim=512,
#     depth=6,
#     heads=8,
#     mlp_dim=2048,
#     dropout=0.1,
#     emb_dropout=0.1,
# )  # 12.817482

# total_params = sum(p.numel() for p in model.parameters()) / 1e6
total_params = (
    sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
)  # Training params

print(total_params)
