import torch

def run():
    # Input tensor
    input_tensor = torch.tensor([[-0.0881,  0.0748,  0.0671],
        [-0.0880,  0.0755,  0.0672],
        [-0.0880,  0.0755,  0.0671],
        [-0.0882,  0.0757,  0.0669],
        [-0.0877,  0.0786,  0.0669],
        [-0.0882,  0.0750,  0.0668],
        [-0.0881,  0.0761,  0.0668],
        [-0.0880,  0.0762,  0.0669],
        [-0.0881,  0.0762,  0.0668],
        [-0.0882,  0.0751,  0.0671],
        [-0.0880,  0.0754,  0.0672],
        [-0.0880,  0.0765,  0.0671],
        [-0.0881,  0.0752,  0.0671],
        [-0.0881,  0.0754,  0.0671],
        [-0.0881,  0.0753,  0.0669],
        [-0.0881,  0.0751,  0.0670],
        [-0.0881,  0.0760,  0.0669],
        [-0.0881,  0.0753,  0.0670],
        [-0.0881,  0.0748,  0.0672],
        [-0.0881,  0.0749,  0.0671],
        [-0.0882,  0.0755,  0.0670],
        [-0.0881,  0.0753,  0.0670],
        [-0.0879,  0.0763,  0.0669],
        [-0.0879,  0.0778,  0.0673],
        [-0.0877,  0.0793,  0.0664],
        [-0.0882,  0.0746,  0.0672],
        [-0.0882,  0.0750,  0.0670],
        [-0.0882,  0.0750,  0.0670],
        [-0.0882,  0.0766,  0.0670],
        [-0.0881,  0.0757,  0.0670],
        [-0.0881,  0.0763,  0.0671],
        [-0.0881,  0.0756,  0.0669]])
    print(f"{input_tensor=} {input_tensor.size()}")

    # Indices tensor
    indices = torch.tensor([[2],
        [2],
        [2],
        [0],
        [1],
        [0],
        [1],
        [1],
        [1],
        [2],
        [2],
        [1],
        [2],
        [2],
        [2],
        [2],
        [0],
        [2],
        [2],
        [2],
        [2],
        [2],
        [1],
        [1],
        [1],
        [2],
        [2],
        [2],
        [1],
        [2],
        [1],
        [0]])

    # Gather values
    output_tensor = input_tensor.gather(dim=1, index=indices)
    print(output_tensor)

if __name__ == "__main__":
    run()