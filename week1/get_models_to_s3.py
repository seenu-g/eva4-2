import torch

from torchvision.models import resnet
model = resnet.resnet34(pretrained=True)
model.eval()
traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224)) 
traced_model.save('resnet34.pt')''

from torchvision.models import mobilenet_v2
model = mobilenet_v2(pretrained=True)
model.eval()
traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224)) 
traced_model.save('mobilenet_v2.pt')

''' use this command to upload to S3
aws s3 mv mobilenet_v2.pt s3://eva4-2/artifacts/week1/
aws s3 mv resnet34.pt s3://eva4-2/artifacts/week1/
'''
