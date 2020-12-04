import torch
import torch.nn as nn
import argparse
from typing import Dict, NamedTuple
from collections import namedtuple
torch.classes.load_library("build/libcustom_class.so")
import time
print(torch.classes.loaded_libraries)

class ModelInputs(NamedTuple):
    inp1 : torch.Tensor
    inp2 : torch.Tensor


class Block(nn.Module):
    def __init__(self, features, num_layers):
        super().__init__()
        layers = [nn.Linear(features, features) for _ in range(num_layers)]
        self.fc_block = nn.Sequential(*layers)

    def forward(self, inp):
        return self.fc_block(inp)

class Net(nn.Module):
    def __init__(self, features, use_plugin):
        super().__init__()
        if use_plugin:
            self.plugin_class = torch.classes.plugin_class.MyLaunchClass("./seq_multi_block_no_plugin.pth")
        else:
            self.plugin_class = torch.classes.plugin_class.MyLaunchClass("");
        self.use_plugin = use_plugin
        self.block1 = Block(features, 10)
        self.block2 = Block(features, 10)
    
    def forward(self, inputs : ModelInputs):
        if self.use_plugin:
            return self.forward_plugin(inputs)
        else:
            return self.forward_noplugin(inputs)

    def forward_noplugin(self, inputs : ModelInputs):
        inp1 = inputs.inp1
        inp2 = inputs.inp2
        out1 = self.block1(inp1)
        out2 = self.block2(inp2)
        return out1 + out2

    def forward_plugin(self, inputs: ModelInputs):
        out1, out2 = self.plugin_class.multi_launch(inputs.inp1, inputs.inp2)
        return out1 + out2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential model with multiple blocks to export and use plugin")
    parser.add_argument('--export_jit', action='store_true',  help="export the model")
    parser.add_argument('--use_plugin', action='store_true',  help="use torchscript plugin")
    args = parser.parse_args()
    export_jit = args.export_jit
    use_plugin = args.use_plugin

    inp = torch.randn((512,512), device="cuda")
    inputs = ModelInputs(inp1 = inp,
                        inp2 = inp)
    net = Net(512, use_plugin)
    net.cuda().eval()

    if export_jit:
        print("Jitting the model")
        net = torch.jit.script(net)
        if use_plugin:
            print("Saving the model: seq_multi_block_w_plugin.pth")
            torch.jit.save(net, "seq_multi_block_w_plugin.pth")
            net = torch.jit.load("./seq_multi_block_w_plugin.pth")
        else:
            print("Saving the model: seq_multi_block_no_plugin.pth")
            torch.jit.save(net, "seq_multi_block_no_plugin.pth")
            net = torch.jit.load("./seq_multi_block_no_plugin.pth")

    print("Running Network with plugin: ", use_plugin)
    for _ in range(10):
        start = time.time()
        out = net(inputs)
        torch.cuda.synchronize()
        print((time.time() - start) * 1000, "ms")


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
    module = torch.jit.load("./seq_multi_block.pth")
    for name,_ in module.named_modules():
        print(name)
    '''

    print("done")



