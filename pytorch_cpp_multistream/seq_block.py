import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, features, num_layers):
        super().__init__()
        layers = [nn.Linear(features, features) for _ in range(num_layers)]
        self.fc_block = nn.Sequential(*layers, 
                                      nn.ReLU())

    def forward(self, inp):
        out = self.fc_block(inp)
        out = self.fc_block(out)
        out = self.fc_block(out)
        out = self.fc_block(out)
        out = self.fc_block(out)
        out = self.fc_block(out)
        out = self.fc_block(out)
        out = self.fc_block(out)
        out = self.fc_block(out)
        out = self.fc_block(out)
        return out

if __name__ == "__main__":
    net = torch.jit.script(Net(64, 10))
    inp = torch.randn((64,64), device="cuda")
    net.cuda().eval()
    net(inp)
    torch.jit.save(net, "seq_block.pth")
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


    ### Load jitted module and print named modules
    '''
    module = torch.jit.load("./seq_block.pth")
    for name,_ in module.named_modules():
        print(name)
    '''

    print("done")



