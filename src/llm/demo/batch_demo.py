from llm.configs.gpt_config import GPT_CONFIG_124M
from llm.demo.dummy_batch import get_dummy_batch
from llm.placeholders import DummyGPTModel


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
