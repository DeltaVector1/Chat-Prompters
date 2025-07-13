"""
KTO strategies for mistral chat template
"""

# pylint: disable=duplicate-code


def argilla(
    cfg,
    **kwargs,
):  # pylint: disable=possibly-unused-variable,unused-argument
    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"<s>[SYSTEM_PROMPT] {sample['system']}[/SYSTEM_PROMPT]"
                f"[INST] {sample['instruction']}[/INST]"
            )
        else:
            sample["prompt"] = (
                f"<s>[INST] {sample['instruction']}[/INST]"
            )
        sample["completion"] = f" {sample['completion']}</s>"
        return sample

    return transform_fn


def argilla_chat(
    cfg,
    **kwargs,
):  # pylint: disable=possibly-unused-variable,unused-argument
    """
    for argilla/kto-mix-15k conversations
    """

    def transform_fn(sample):
        sample["prompt"] = (
            f"<s>[INST] {sample['completion'][0]['content']}[/INST]"
        )
        sample["completion"] = f" {sample['completion'][1]['content']}</s>"
        return sample

    return transform_fn


def intel(cfg, **kwargs):  # pylint: disable=possibly-unused-variable,unused-argument
    """
    For Intel Orca KTO
    ex: argilla/distilabel-intel-orca-kto
    """

    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"<s>[SYSTEM_PROMPT] {sample['system']}[/SYSTEM_PROMPT]"
                f"[INST] {sample['question']}[/INST]"
            )
        else:
            sample["prompt"] = (
                f"<s>[INST] {sample['question']}[/INST]"
            )
        sample["completion"] = f" {sample['completion']}</s>"
        return sample

    return transform_fn


def prompt_pairs(
    cfg, **kwargs
):  # pylint: disable=possibly-unused-variable,unused-argument
    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"<s>[SYSTEM_PROMPT] {sample['system']}[/SYSTEM_PROMPT]"
                f"[INST] {sample['prompt']}[/INST] "
            )
        else:
            sample["prompt"] = (
                f"<s>[INST] {sample['prompt']}[/INST]"
            )
        sample["completion"] = f" {sample['completion']}</s>"
        return sample

    return transform_fn


def ultra(cfg, **kwargs):  # pylint: disable=possibly-unused-variable,unused-argument
    """
    for ultrafeedback binarized conversations
    ex: argilla/ultrafeedback-binarized-preferences-cleaned-kto
    """

    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"<s>[SYSTEM_PROMPT] {sample['system']}[/SYSTEM_PROMPT]"
                f"[INST] {sample['prompt']}[/INST]"
            )
        else:
            sample["prompt"] = (
                f"<s>[INST] {sample['prompt']}[/INST]"
            )
        sample["completion"] = f" {sample['completion']}</s>"
        return sample

    return transform_fn
