import torch

x = torch.randn((512, 512), device='cuda')
y = torch.randn((512, 512), device='cuda')
for _ in range(10):
    z = torch.matmul(x,y)
    w = torch.matmul(x,y)
s = torch.cuda.Stream()
torch.cuda.profiler.start()
for _ in range(20):
    z = torch.matmul(x,y)
    with torch.cuda.stream(s):
        w = torch.matmul(x,y)
torch.cuda.synchronize()
torch.cuda.profiler.stop()



