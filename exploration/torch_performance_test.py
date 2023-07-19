
import torch

if __name__ == "__main__":
    print(f"{torch.cuda.is_available()=}")
    print(f"{torch.cuda.current_device()=}")


    def batched_dot_mul_sum(a, b):
        '''Computes batched dot by multiplying and summing'''
        return a.mul(b).sum(-1)


    def batched_dot_bmm(a, b):
        '''Computes batched dot by reducing to ``bmm``'''
        a = a.reshape(-1, 1, a.shape[-1])
        b = b.reshape(-1, b.shape[-1], 1)
        return torch.bmm(a, b).flatten(-3)


    # Input for benchmarking
    x = torch.randn(10000, 64, device=torch.device("cuda"))
    print(x.get_device())


    # Ensure that both functions compute the same output
    assert batched_dot_mul_sum(x, x).allclose(batched_dot_bmm(x, x))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for i in range(1000):
        _ = batched_dot_mul_sum(x, x)
    end.record()
    torch.cuda.synchronize()


    print(f"CUDA elapsed: {start.elapsed_time(end):>5.1f} ms")