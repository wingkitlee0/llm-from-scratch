import torch

from llm import create_dataloader_v1


def main():
    with open("data/the-verdict.txt", "r") as f:
        raw_data = f.read()

    max_length = 4
    dataloader = create_dataloader_v1(
        raw_data, batch_size=8, max_length=max_length, stride=max_length
    )
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)

    vocab_size = 50257
    output_dim = 256

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    inputs, targets = first_batch

    token_embeddings = token_embedding_layer(inputs)

    print(f"{token_embeddings.shape=}")

    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))

    print(f"{pos_embeddings.shape=}")

    input_embeddings = token_embeddings + pos_embeddings
    print(f"{input_embeddings.shape=}")


if __name__ == "__main__":
    main()
