import torch
import plugins
import time

#inp = torch.randn(1, 3, 224, 224)
inp = torch.rand((512, 512), device='cuda')
for _ in range(10):
    plugins.launch(inp)
    #plugins.multi_launch(inp)

for _ in range(20):
    start = time.time()
    out1, out2 = plugins.launch(inp)
    #mout1, mout2 = plugins.multi_launch(inp)
    print("Python time: ", (time.time() - start)*1000)
