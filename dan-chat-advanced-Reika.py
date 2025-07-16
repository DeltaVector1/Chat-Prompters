"""Reka Flash 3.1 style prompt tokenizing strategy"""

import copy
import logging
from collections import defaultdict
from typing import Generator, List, Tuple, Dict

from axolotl.prompt_tokenizers import (
    PromptTokenizingStrategy,
    parse_tokenized_to_result,
    tokenize_prompt_default,
)

LOG = logging.getLogger("axolotl")

IGNORE_TOKEN_ID = -100

# Reka Flash 3.1 format - no special tokens, just plain text
turn_separator = "<sep>"

# Simple role prefixes
system_prefix = "human: "  # System prompt gets merged with first user turn
user_prefix = "human: "
assistant_prefix = "assistant: "
tool_prefix = "human: "  # Treat tool calls as user input in this format

class RekaFlashPrompter:
    """Handles the Reka Flash 3.1 prompt format"""
    
    def __init__(self):
        self.system_prompt = None
    
    def build_prompt(self, conversations: List[Dict]) -> Generator[Tuple[str, str, bool, str], None, None]:
        """Build prompt in Reka Flash format"""
        
        for i, turn in enumerate(conversations):
            role = turn["from"]
            message = turn["value"]
            
            # Handle system prompt - prepend to first user message
            if i == 0 and self.system_prompt:
                if role in ["user", "human"]:
                    message = f"{self.system_prompt} {message}"
                else:
                    # If first turn isn't user, add system as separate turn
                    yield "system", self.system_prompt, False, ""
            
            # Map roles to our format
            if role in ["user", "human"]:
                yield "user", message, False, ""
            elif role == "assistant":
                yield "assistant", message, True, ""
            elif role == "system":
                if i > 0:  # Only yield if not already handled above
                    yield "system", message, False, ""
            elif role == "tool":
                yield "tool", message, False, ""

class RekaFlashTokenizingStrategy(PromptTokenizingStrategy):
    """Tokenizing strategy for Reka Flash 3.1 format"""
    
    def __init__(self, prompter, tokenizer, cfg=None, max_length=2048):
        super().__init__(prompter, tokenizer, cfg, max_length)
        self.sequence_len = max_length
    
    def tokenize_prompt(self, prompt):
        prompt_parts = list(self.prompter.build_prompt(prompt["conversations"]))
        tokenized_parts = []
        total_length = 0
        
        for role, message, loss, prefix in prompt_parts:
            # Build the actual text for this turn
            if role == "system":
                # System gets merged with user, so skip standalone system turns
                continue
                
            role_prefix = ""
            if role == "user":
                role_prefix = user_prefix
            elif role == "assistant":
                role_prefix = assistant_prefix
            elif role == "tool":
                role_prefix = tool_prefix
            
            # Add turn separator after each message except the last
            full_message = f"{role_prefix}{message}"
            if role != "assistant":  # Don't add sep after assistant responses
                full_message += turn_separator
            
            # Tokenize this part
            res = self.tokenizer(
                full_message,
                add_special_tokens=False,  # Reka uses no special tokens
                return_tensors=None
            )
            
            labels = [IGNORE_TOKEN_ID] * len(res["input_ids"])
            if loss:
                # Only train on assistant responses
                labels = copy.deepcopy(res["input_ids"])
            
            part_length = len(res["input_ids"])
            if total_length + part_length > self.sequence_len:
                break
            
            tokenized_parts.append({
                "input_ids": res["input_ids"],
                "attention_mask": res["attention_mask"],
                "labels": labels,
                "role": role,
                "loss": loss
            })
            total_length += part_length
        
        # Clean up trailing non-assistant turns
        while tokenized_parts and (tokenized_parts[-1]["role"] in ["user", "system"] or not tokenized_parts[-1]["loss"]):
            tokenized_parts.pop()
        
        # Combine all parts
        result = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        
        for part in tokenized_parts:
            result["input_ids"].extend(part["input_ids"])
            result["attention_mask"].extend(part["attention_mask"])
            result["labels"].extend(part["labels"])
        
        return result
