#!/usr/bin/env python3
import os
import yaml
import json

import argparse
import logging

import random
import re
import time

import tempfile

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import Table

from concurrent.futures import ThreadPoolExecutor

from rich.progress import Progress

from colorama import Fore, Style, init

# Set up logging to the console
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Constants for the DanChatML format
format_tokens = {
    "starting_sequence": {
        "text": "",
        "tokens": None
    },
    "system_prefix": {
        "text": "<|im_start|>system\n",
        "tokens": None
    },
    "user_prefix": {
        "text": "<|im_start|>user\n",
        "tokens": None
    },
    "assistant_prefix": {
        "text": "<|im_start|>assistant\n",
        "tokens": None
    },
    "tool_prefix": {
        "text": "<|im_start|>tool\n",
        "tokens": None
    },
    "turn_seperator": {
        "text": "\n",
        "tokens": None
    },
    "turn_suffix": {
        "text": "<|im_end|>",
        "tokens": None
    }
}

mask_token_id = -100

system_from_values = ["system", "sys"]
user_from_values = ["user", "human"]
assistant_from_values = ["assistant", "model", "gpt"]
tool_from_values = ["tool", "function"]

# Dan Chat Advanced dataset format:
"""
# Example input format for DanChatML dataset:
[
    {
        "conversations": [
            {
            "from": "system",      # Indicates system message role
            "value": "System message content here"  # Content of the system message
            },
            {
            "from": "user",        # Indicates user message role
            "value": "User message content here"    # Content of user message
            "loss": true           # Whether to include in loss calculation
            },
            {
            "from": "model",       # Indicates model/assistant response
            "prefix": "Optional prefix text that won't be trained on",  # Prefix excluded from training
            "value": "Assistant response here",     # Main response content
            "loss": false          # This response not included in loss
            },
            {
            "from": "tool",        # Indicates tool output
            "value": "Tool output here"    # Content from tool execution
            },
            {
            "from": "model",       # Another model response
            "prefix": "Optional prefix text that won't be trained on",  # Prefix text
            "value": "The actual model response"    # Main model response
            }
        ]
    },
    ...  # Additional conversation examples
]
"""

# How that would look like formatted for DanChatML in plain text:
"""
[gMASK]<sop><|system|>System message content here<|endoftext|><|user|>User message content here<|endoftext|><|assistant|>Optional prefix text that won't be trained onAssistant response here<|endoftext|><|observation|>Tool output here<|endoftext|><|assistant|>Optional prefix text that won't be trained onThe actual model response<|endoftext|>
"""

# The output format for the tokenized data:
"""
[
    {
        "input_ids": [271, 299, 99],
        "attention_mask": [1, 1, 1],
        "labels": [271, -100, 99]
    },
    {
        "input_ids": [87, 227, 8383, 12],
        "attention_mask": [1, 1, 1, 1],
        "labels": [87, 227, 8383, 12]
    }
]
"""

# Sample from a yaml file defining the datasets, model, and other parameters
# Example YAML config:
"""
base_model: /run/media/pocketdoc/extended-storage/Models/Dans-DiscountModels_mistral-7b-v0.3-ChatML
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

trust_remote_code:

# wandb configuration
wandb_project: 7b-m-dans-repremover
wandb_watch:

wandb_run_id: V1.3.0-1-1 # V{Version}-{Run Number}-{Attempt Number}
wandb_log_model:

# dataset settings (local or huggingface repo)
datasets:
  - path: PocketDoc/Dans-Systemmaxx
    type: dan-chat-advanced
  # Personality and Character Development
  - path: PocketDoc/Dans-Personamaxx-VN
    type: dan-chat-advanced

sequence_len: 32768
"""

# Function to load the YAML configuration file and return the model, max length, and datasets
def load_yaml_config(yaml_file):
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)

    model = config.get("base_model")
    max_length = config.get("sequence_len", 512)
    datasets = [d["path"] for d in config.get("datasets", []) if d.get("type") == "dan-chat-advanced"]
    wandb_project = config.get("wandb_project", "default_project")
    wandb_run_id = config.get("wandb_run_id", "default_run_id")

    return {"model": model, "max_length": max_length, "datasets": datasets, "wandb_project": wandb_project, "wandb_run_id": wandb_run_id}

def fill_format_tokens(tokenizer):
    """
    Fill the format tokens with their tokenized values if the current value is None.
    """
    # Handle the turn_seperator specially by combining suffix + separator + prefix
    # and removing the suffix and prefix tokens from the result
    for key, value in format_tokens.items():
        if key != "turn_seperator" and value["tokens"] is None:
            # Standard tokenization for non-separator tokens
            tokens = tokenizer.encode(value["text"], add_special_tokens=False)
            format_tokens[key]["tokens"] = tokens
            logger.debug(f"Tokenized {key}: {value['text']} -> {tokens}")

    # Special handling for the turn separator
    if format_tokens["turn_seperator"]["tokens"] is None:
        # Create a combined string of suffix + separator + prefix
        combined_text = (
            format_tokens["turn_suffix"]["text"] +
            format_tokens["turn_seperator"]["text"] +
            format_tokens["assistant_prefix"]["text"]
        )

        # Tokenize the combined text
        combined_tokens = tokenizer.encode(combined_text, add_special_tokens=False)

        # Get the individual component tokens
        suffix_tokens = format_tokens["turn_suffix"]["tokens"]
        prefix_tokens = format_tokens["assistant_prefix"]["tokens"]

        # Check that the starting ids of the combined tokens match with the suffix ids
        if combined_tokens[:len(suffix_tokens)] != suffix_tokens:
            raise ValueError("Tokenization failed: combined tokens do not match suffix tokens.")

        # Check that the ending ids of the combined tokens match with the prefix ids
        if combined_tokens[-len(prefix_tokens):] != prefix_tokens:
            raise ValueError("Tokenization failed: combined tokens do not match prefix tokens.")

        # Remove the suffix and prefix tokens from the combined tokens
        separator_tokens = combined_tokens[len(suffix_tokens):-len(prefix_tokens)]

        # Store the separator tokens
        format_tokens["turn_seperator"]["tokens"] = separator_tokens

        logger.debug(f"Tokenized turn_seperator: {format_tokens['turn_seperator']['text']} -> {separator_tokens}")

def normalize_conversation(item):
    """
    Normalize the conversation item in the following way:
        1. Normalize "from" values to standard ones: "system", "human", "gpt", "tool"
        2. Add "loss" key if missing with appropriate default (true for gpt, false otherwise)

    Args:
        item: A list of turns

    Returns:
        Normalized conversation item
    """
    for turn in item:
        # Normalize the "from" field
        from_value = turn["from"].lower()

        if from_value in system_from_values:
            turn["from"] = "system"
        elif from_value in user_from_values:
            turn["from"] = "human"
        elif from_value in assistant_from_values:
            turn["from"] = "gpt"
        elif from_value in tool_from_values:
            turn["from"] = "tool"

        # Add the "loss" key if it doesn't exist
        if "loss" not in turn or turn["loss"] is None:
            # Only gpt turns have loss=True by default
            turn["loss"] = (turn["from"] == "gpt")

    return item


def tokenize_item(item, tokenizer, max_length):
    """
    Tokenize a conversation item with proper formatting for Dan Chat ML.

    Args:
        item: A conversation item with 'conversations' field
        tokenizer: The HuggingFace tokenizer
        max_length: Maximum sequence length

    Returns:
        Dictionary with 'input_ids', 'attention_mask', and 'labels'
    """
    # Check that the format tokens are filled
    for key, value in format_tokens.items():
        if value["tokens"] is None:
            raise ValueError(f"Tokenization failed for {key}: {value['text']}")

    # Get the turns from the item
    turns = normalize_conversation(item["conversations"])

    # Initialize the conversation tokens
    starting_sequence_tokens = format_tokens["starting_sequence"]["tokens"]
    input_ids = list(starting_sequence_tokens)  # Use the defined starting sequence tokens
    labels = [mask_token_id] * len(starting_sequence_tokens)  # Mask the starting sequence tokens

    # Iterate through the turns in the conversation and add a key with a dict containing the input_ids and labels for that turn, stop when the sum of the lengths of the input_ids is greater than the max_length when added to the length of the input_ids var
    # Temporary storage for turns
    turns_data = []

    for turn_idx, turn in enumerate(turns):
        # Initialize the tokenized turn data
        turn_tokenized = {
            "loss": turn.get("loss", False),  # Default to False if loss not specified
            "input_ids": [],
            "labels": []
        }

        # Determine the turn role
        from_value = turn["from"].lower()

        # Check if this turn should contribute to loss calculation
        should_contribute_to_loss = turn.get("loss", False)

        # Select the appropriate role tokens based on the turn's role
        role_tokens = None
        if from_value in system_from_values:
            role_tokens = format_tokens["system_prefix"]["tokens"]
        elif from_value in user_from_values:
            role_tokens = format_tokens["user_prefix"]["tokens"]
        elif from_value in assistant_from_values:
            role_tokens = format_tokens["assistant_prefix"]["tokens"]
        elif from_value in tool_from_values:
            role_tokens = format_tokens["tool_prefix"]["tokens"]
        else:
            # Raise an error if the role is not recognized
            raise ValueError(f"Unknown role '{from_value}' in turn {turn_idx}.")

        # Add the role tokens to the input_ids and labels masked
        turn_tokenized["input_ids"].extend(role_tokens)
        turn_tokenized["labels"].extend([mask_token_id] * len(role_tokens))

        # Get the turn value
        value = turn["value"]

        # Check if it is a string of non zero length
        if (not isinstance(value, str) or len(value) == 0) and from_value in assistant_from_values:
            # Skip turns with empty or non-string values
            logger.debug(f"Skipping turn {turn_idx} with invalid value: {value} in:\n{json.dumps(item, indent=4)}")
            return None

        # Check if there's a prefix for this turn
        turn_prefix = turn.get("prefix", "")

        # Tokenize the prefix
        if turn_prefix != "" and turn_prefix is not None:
            prefix_tokens = tokenizer.encode(turn_prefix, add_special_tokens=False)
            turn_tokenized["input_ids"].extend(prefix_tokens)
            turn_tokenized["labels"].extend([mask_token_id] * len(prefix_tokens))

        value_tokens = tokenizer.encode(value, add_special_tokens=False)

        turn_tokenized["input_ids"].extend(value_tokens)

        if should_contribute_to_loss:
            turn_tokenized["labels"].extend(value_tokens)
        else:
            turn_tokenized["labels"].extend([mask_token_id] * len(value_tokens))

        # Add the turn suffix token
        turn_tokenized["input_ids"].extend(format_tokens["turn_suffix"]["tokens"])

        if should_contribute_to_loss:
            turn_tokenized["labels"].extend(format_tokens["turn_suffix"]["tokens"])
        else:
            turn_tokenized["labels"].extend([mask_token_id] * len(format_tokens["turn_suffix"]["tokens"]))

        # Check if adding this turn would exceed max_length
        # Calculate the length of all current tokens plus the new turn (and separator if needed)
        separator_length = len(format_tokens["turn_seperator"]["tokens"]) if turns_data else 0

        # Calculate the total length if we were to add this turn
        current_length = len(input_ids)
        new_turn_length = len(turn_tokenized["input_ids"])

        # If we would exceed the max length, break out of the loop
        if current_length + separator_length + new_turn_length > max_length:
            break

        # Add the turn to our temporary storage
        turns_data.append(turn_tokenized)

    # Check if any of the turns should contribute to loss, if not, return None
    if not any(turn["loss"] for turn in turns_data):
        return None # Return None to indicate this item should be discarded

    # Remove any turns after the last turn that contributes to loss
    last_loss_turn = max(i for i, turn in enumerate(turns_data) if turn["loss"])
    turns_data = turns_data[:last_loss_turn + 1]

    # Add the turns to the input_ids and labels with the separator between them
    for turn_data in turns_data:
        input_ids.extend(turn_data["input_ids"])
        labels.extend(turn_data["labels"])

        # Add the separator if this is not the last turn
        if turn_data != turns_data[-1]:
            input_ids.extend(format_tokens["turn_seperator"]["tokens"])
            labels.extend([mask_token_id] * len(format_tokens["turn_seperator"]["tokens"]))

    # Create the attention mask
    attention_mask = [1] * len(input_ids)

    # del everything we don't need anymore
    del turns_data

    # Return the tokenized item
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def tokenize_dataset(dataset, tokenizer, max_length):
    """
    Tokenize a dataset with proper formatting for Dan Chat ML.

    Args:
        dataset: A HF dataset id
        tokenizer: The HuggingFace tokenizer
        max_length: Maximum sequence length

    Returns:
        List of tokenized items
    """

    # Load the dataset in streaming mode to handle potentially huge datasets
    # This avoids loading the entire dataset into memory at once, preventing
    # the pyarrow offset overflow error.
    logger.info(f"Loading dataset {dataset} in streaming mode...")
    try:
        # Try loading with trust_remote_code=True if needed for this dataset
        loaded_data = load_dataset(dataset, split="train", streaming=True, trust_remote_code=True)
        # You might need trust_remote_code=True if the dataset has a custom loading script
    except Exception as e:
        logger.warning(f"Failed loading {dataset} with trust_remote_code=True, trying without: {e}")
        loaded_data = load_dataset(dataset, split="train", streaming=True)

    logger.info(f"Successfully loaded {dataset} stream.")

    tokenized_items = []

    # Tokenize each item in the dataset using concurrent processing
    # The ThreadPoolExecutor works fine with the streaming dataset iterator
    # Note: Progress bars might not show total count with streaming datasets easily
    # Consider adding manual progress logging if needed.
    processed_count = 0
    start_time = time.time()
    log_interval = 10000 # Log progress every N items

    with ThreadPoolExecutor() as executor:
        futures = []
        # Iterate directly over the streaming dataset
        for item in loaded_data:
            # Submit the tokenization task to the executor
            future = executor.submit(tokenize_item, item, tokenizer, max_length)
            futures.append(future)

            # Process results periodically to avoid holding too many futures in memory
            # and to provide progress feedback
            if len(futures) >= executor._max_workers * 2: # Process batch when futures buffer fills
                completed_futures = []
                while futures:
                    future = futures.pop(0)
                    try:
                        result = future.result()
                        if result is not None:
                            tokenized_items.append(result)
                        processed_count += 1
                        if processed_count % log_interval == 0:
                             elapsed = time.time() - start_time
                             logger.info(f"Processed {processed_count} items from {dataset}... ({elapsed:.2f}s)")
                    except Exception as exc:
                         logger.error(f'An item from {dataset} generated an exception during tokenization: {exc}')
                         # Optionally log the problematic item (can be verbose)
                         # logger.error(f"Problematic item: {item}") # Be careful with large items
                    completed_futures.append(future) # Keep track if needed, or just let them go


        # Process any remaining futures
        logger.info(f"Processing remaining {len(futures)} futures for {dataset}...")
        for future in futures:
             try:
                result = future.result()
                if result is not None:
                    tokenized_items.append(result)
                processed_count += 1
                if processed_count % log_interval == 0:
                     elapsed = time.time() - start_time
                     logger.info(f"Processed {processed_count} items from {dataset}... ({elapsed:.2f}s)")
             except Exception as exc:
                 logger.error(f'An item from {dataset} generated an exception during final tokenization: {exc}')

    total_time = time.time() - start_time
    logger.info(f"Finished processing {processed_count} items from {dataset} in {total_time:.2f}s")

    # Return the tokenized items
    return tokenized_items


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Tokenize a Dan-Chat-Advanced dataset to the Axolotl standard.")
    # Add argument for the YAML configuration file
    parser.add_argument(
        "yaml",
        type=str,
        help="Path to the YAML configuration file.",
    )
    # Add an argument for the output json file
    parser.add_argument(
        "output",
        type=str,
        default="./output.parquet",
        help="Path to the output dataset file.",
    )
    # Add an argument to enable or disable debug mode
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode.",
    )
    # Add an argument to enable reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    # Add an argument to enable pushing to huggingface
    parser.add_argument(
        "--hf-push",
        type=str,
        default=None,
        help="HuggingFace repo to push the dataset to.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Enable debug mode if specified
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled.")
    else:
        logger.setLevel(logging.INFO)

    # If the seed is specified, set the random seed for reproducibility
    if args.seed:
        random.seed(args.seed)
        logger.debug(f"Random seed set to {args.seed}")

    # Load the YAML configuration
    config = load_yaml_config(args.yaml)
    model = config["model"]
    max_length = config["max_length"]
    datasets = config["datasets"]
    wandb_project = config["wandb_project"]
    wandb_run_id = config["wandb_run_id"]
    logger.info(f"Loaded config: {config}")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Tokenize the formatting constants
    fill_format_tokens(tokenizer)

    # Check that all format tokens are filled
    for key, value in format_tokens.items():
        if value["tokens"] is None:
            raise ValueError(f"Tokenization failed for {key}: {value['text']}")
    logger.info("Format tokens filled successfully.")
    logger.debug(f"Format tokens: {format_tokens}")

    # Create a temporary directory for the tokenized data
    temp_dir = tempfile.TemporaryDirectory()
    logger.debug(f"Temporary directory created at {temp_dir.name}")

    # Tokenize each dataset and save the results as parquet files in the temporary directory
    for dataset in datasets:
        logger.info(f"Tokenizing dataset: {dataset}")
        tokenized_data = tokenize_dataset(dataset, tokenizer, max_length)
        logger.info(f"Tokenized {len(tokenized_data)} items from {dataset}.")
        # Log how many tokens there are total
        total_tokens = sum(len(item["input_ids"]) for item in tokenized_data)
        logger.info(f"Total tokens in {dataset}: {total_tokens}")

        # Log how many unmasked tokens there are
        total_unmasked_tokens = sum(len(item["input_ids"]) - item["labels"].count(mask_token_id) for item in tokenized_data)
        logger.info(f"Total unmasked tokens in {dataset}: {total_unmasked_tokens}")

        # Save the tokenized data to a temporary parquet file
        temp_file = os.path.join(temp_dir.name, f"{os.path.basename(dataset)}.parquet")
        table = Table.from_pylist(tokenized_data)
        del tokenized_data
        pq.write_table(table, temp_file, compression="snappy")
        logger.info(f"Tokenized data saved to {temp_file}.")

    # Combine all tokenized data into a single list
    all_tokenized_data = []

    for dataset in datasets:
        temp_file = os.path.join(temp_dir.name, f"{os.path.basename(dataset)}.parquet")
        table = pq.read_table(temp_file)
        all_tokenized_data.extend(table.to_pylist())
        logger.info(f"Loaded tokenized data from {temp_file}.")

    # Shuffle the combined tokenized data
    random.shuffle(all_tokenized_data)

    # If debug mode is enabled, print the ids for the first item
    if args.debug:
        init()  # Initialize colorama

        # Decode and print tokens with color
        input_ids = all_tokenized_data[0]['input_ids']
        labels = all_tokenized_data[0]['labels']

        decoded_tokens = []
        for i, token_id in enumerate(input_ids):
            token = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=True)
            color = Fore.GREEN if labels[i] != mask_token_id else Fore.RED
            decoded_tokens.append(f"{color}{token}{Style.RESET_ALL}")

        logger.debug("Decoded tokens (green=unmasked, red=masked):")
        logger.debug("".join(decoded_tokens))

    # Get the total number of tokens in the combined dataset
    total_tokens = sum(len(item["input_ids"]) for item in all_tokenized_data)
    logger.info(f"Total tokens in combined dataset: {total_tokens}")

    # log how many unmasked tokens there are
    total_unmasked_tokens = sum(len(item["input_ids"]) - item["labels"].count(mask_token_id) for item in all_tokenized_data)
    logger.info(f"Total unmasked tokens in combined dataset: {total_unmasked_tokens}")

    # Save the tokenized data to a compressed parquet file
    output_file = args.output

    logger.info(f"Saving tokenized data to {output_file}...")

    # With these lines:
    table = pa.Table.from_pylist(all_tokenized_data)
    del all_tokenized_data
    pq.write_table(table, output_file, compression="snappy")

    del table

    logger.info(f"Tokenized data saved to {output_file}.")

    if args.hf_push:
        # Push the dataset to HuggingFace
        logger.info(f"Pushing dataset to HuggingFace: {args.hf_push}")
        dataset = Dataset.from_parquet(output_file)
        dataset.push_to_hub(args.hf_push, private=True, token=os.getenv("HF_TOKEN"), max_shard_size="5GB")
        logger.info(f"Dataset pushed to HuggingFace: {args.hf_push}")


# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()
