
import torch
import lightning as L

def setup():
    fabric = L.Fabric(devices=4, accelerator="cpu")
    fabric.launch(main)

# def main(fabric):
#     # Data is different on each process
#     learning_rate = torch.rand(1)
#     print("Before broadcast:", learning_rate)

#     # Transfer the tensor from one process to all the others
#     learning_rate = fabric.broadcast(learning_rate)
#     print("After broadcast:", learning_rate)

# def main(fabric):
#     # Data is different in each process
#     data = torch.tensor(10 * fabric.global_rank)

#     # Every process gathers the tensors from all other processes
#     # and stacks the result:
#     result = fabric.all_gather(data)
#     print("Result of all-gather:", result)  # tensor([ 0, 10, 20, 30])

def main(fabric):
    # Data is different in each process
    data = torch.tensor(10 * fabric.global_rank)

    # Sum the tensors from every process
    result = fabric.all_reduce(data, reduce_op="sum")

    # sum(0 + 10 + 20 + 30) = tensor(60)
    print("Result of all-reduce:", result)


if __name__ == "__main__":
    setup()