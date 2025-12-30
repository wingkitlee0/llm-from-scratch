from llm.configs.gpt_config import GPT_CONFIG_124M
from llm.gpt2.placeholders import DummyGPTModel
from llm.utils import get_dummy_batch


def main():
    batch = get_dummy_batch()

    print(batch.shape)
    print(batch)

    model = DummyGPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print(logits.shape)
    # print(logits)


if __name__ == "__main__":
    main()
