import torch

a = torch.tensor([[7., 7.], [1., 1.]], requires_grad=True)
# g = a.clone()
#
# g[0,1] = a[0,1]/2.0
# # b = a*a
# k = g*g
#
# # c = (b*b).mean()
# d = (k*k).mean()
#
# d.backward()
# print(23)
# # c.backward()
#
# # b[0,0] = a[0,0]
# # print(b.data_ptr())
# # print(b.data_ptr())
# # c.backward()
# # print(100)


b = a.clone() * a.clone()
b[0,0] = a[0,1] *a[1,0]
c = (b*b).mean()
c.backward()
print(100)



(segmap.bool()[:,0]).unsqueeze(0).shape
gamma_clone.masked_scatter_()

(segmap.bool()[:,0])