import torch


def main():
    inputs = torch.rand(6, 3)  # 6x of size-3 vectors

    query = inputs[1]

    attn_scores_2 = torch.empty(inputs.shape[0])  # 6 elements

    for i, x in enumerate(inputs):
        attn_scores_2[i] = torch.dot(query, x)

    print(f"{attn_scores_2=}")
    print(f"{attn_scores_2.shape=}")

    # Normalize the attention scores
    attn_scores_2_norm = attn_scores_2 / attn_scores_2.sum()

    print(f"{attn_scores_2_norm=}")

    print(f"{attn_scores_2_norm.sum()=}")


if __name__ == "__main__":
    main()
