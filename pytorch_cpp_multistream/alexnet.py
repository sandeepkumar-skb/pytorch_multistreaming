import torch
import torchvision.models as models

alexnet = torch.jit.script(models.alexnet(pretrained=True))

alexnet.eval()
inp = torch.randn(1, 3, 224, 224)
output = alexnet(inp)
torch.jit.save(alexnet, "alexnet_scripted.pth")
print("Done!")

