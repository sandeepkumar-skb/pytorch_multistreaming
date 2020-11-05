import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, features, num_layers):
        super().__init__()
        layers = [nn.Linear(features, features) for _ in range(num_layers)]
        self.fc_block = nn.Sequential(*layers)

    def forward(self, inp):
        return self.fc_block(inp)

class Net(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.block1 = Block(features, 10)
        self.block2 = Block(features, 10)
        self.block3 = Block(features, 10)
        self.block4 = Block(features, 10)
    
    def forward(self, inp):
        out1 = self.block1(inp)
        out2 = self.block2(inp)
        out3 = self.block3(inp)
        out4 = self.block4(inp)
        return (out1 + out2 + out3 + out4)



if __name__ == "__main__":
    net = torch.jit.script(Net(512))
    inp = torch.randn((512,512), device="cuda")
    net.cuda().eval()
    net(inp)
    torch.jit.save(net, "seq_multi_block.pth")

    ### Profiling Start
    '''
    for _ in range(10):
        net(inp)
    torch.cuda.synchronize()
    #torch.cuda.profiler.start()
    for _ in range(50):
        net(inp)
    torch.cuda.synchronize()
    #torch.cuda.profiler.stop()
    '''
    ### Profiling End

    ### Load jitted module and print named_modules
    '''
    module = torch.jit.load("./seq_block.pth")
    for name,_ in module.named_modules():
        print(name)
    '''

    print("done")



