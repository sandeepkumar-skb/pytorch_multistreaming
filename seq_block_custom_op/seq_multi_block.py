import torch
import torch.nn as nn
import argparse
torch.ops.load_library("build/libplugins.so")

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
    
    def forward(self, inp):
        out1 = self.block1(inp)
        out2 = self.block2(inp)
        return out1 + out2

    @torch.jit.export
    def forward_plugin(self, inp):
        out1, out2 = torch.ops.plugins.multi_launch(inp)
        return out1 + out2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential model with multiple blocks to export and use plugin")
    parser.add_argument('--export_jit', action='store_true',  help="export the model")
    parser.add_argument('--use_plugin', action='store_true',  help="use torchscript plugin")
    args = parser.parse_args()
    export_jit = args.export_jit
    use_plugin = args.use_plugin

    inp = torch.randn((512,512), device="cuda")
    net = Net(512)
    net.cuda().eval()

    if export_jit:
        print("Jitting the model")
        net = torch.jit.script(net)
    if use_plugin:
        net.forward_plugin(inp)
    else:
        net(inp)

    #torch.jit.save(net, "seq_multi_block.pth")

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



