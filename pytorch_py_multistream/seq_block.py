import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, features):
        super().__init__()
        layers = [nn.Linear(features, features) for _ in range(10)]
        self.fc_block = nn.Sequential(*layers)

    def forward(self, inp):
        return self.fc_block(inp)


if __name__ == "__main__":
    #net = torch.jit.script(Net(10))
    net1 = Net(512)
    net2 = Net(512)
    inp = torch.randn((512,512), device="cuda")
    #torch.jit.save(net, "dummy_net.pth")
    net1.cuda().eval()
    net2.cuda().eval()
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    for _ in range(10):
        net1(inp)
        net2(inp)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for _ in range(50):
        with torch.cuda.stream(s1):
            net1(inp)
        with torch.cuda.stream(s2):
            net2(inp)
    #with torch.cuda.stream(s2):
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print("done")



