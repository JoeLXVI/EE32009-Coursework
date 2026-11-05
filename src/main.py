import scipy.io as spio

# Import MATLAB Dataset
mat = spio.loadmat("D1.mat", squeeze_me=True)
d = mat["d"]
Index = mat["Index"]
Class = mat["Class"]
