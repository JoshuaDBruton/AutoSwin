import torch.nn as nn
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model, TaskType


class PEFTTextEncoder(nn.Module):
    def __init__(
        self,
        base_text_encoder,
        num_virtual_tokens=144,
        prompt_tuning_init=PromptTuningInit.RANDOM
    ):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens

        peft_config = PromptTuningConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            num_virtual_tokens=num_virtual_tokens,
            prompt_tuning_init=prompt_tuning_init,
        )

        base_text_encoder.device = base_text_encoder.transformer.device

        self.peft_model = get_peft_model(base_text_encoder.transformer, peft_config)

        self.proj = getattr(base_text_encoder, 'proj', None)

        if self.proj is None:
            raise Exception("Panic! Base Text Encoder does not have projection.")

    def forward(self, texts):
        # Input should already be tokenized, not using peft tokenizer
        inputs = texts

        outputs = self.peft_model(inputs)

        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        elif hasattr(outputs, 'pooler_output'):
            return outputs.pooler_output
        else:
            raise Exception("Panic! Hidden state not found check text encoder parameters to find proper name.")
            # hidden_states = outputs[0]

        cls_embeds = hidden_states[:, self.num_virtual_tokens, :]

        if self.proj is not None:
            final_embeds = self.proj(cls_embeds)
        else:
            final_embeds = cls_embeds

        return final_embeds

