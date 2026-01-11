from llm.placeholders import DummyGPTModel

from llm.configs.gpt_config import DEFAULT_GPT_CONFIG
from llm.demo.dummy_batch import get_dummy_batch


def main():
    batch = get_dummy_batch()

    print(batch.shape)
    print(batch)

    model = DummyGPTModel(DEFAULT_GPT_CONFIG)
    logits = model(batch)
    print(logits.shape)
    # print(logits)


if __name__ == "__main__":
    main()
