import torch

def conditional_indexing():
    actions = torch.tensor([[1], [1], [2], [2]], dtype=torch.int)
    share = torch.tensor([[0], [1], [0], [1]], dtype=torch.int)
    print(actions)
    print(share)

    idx = ((actions == 1) | ((actions == 2) & (share == 1)))
    print(idx)

def run():
    a = torch.tensor(
        [[313.6887],
         [312.3073],
         [304.4344],
         [297.3474]])
    b = torch.tensor(
        [[301.3113],
         [300.8559],
         [295.5355],
         [293.8388]])
    print(a.shape, b.shape)

    r = torch.div(a, b)
    print(r)

if __name__ == "__main__":
    T = torch.randint(0, 3, size=(4, 20),  dtype=torch.int64)
    print(T.size()[0], T)