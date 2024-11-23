"""
@SUN, Haoran  2024/11/24
Since the authors were training in a distributed manner,
the parameters in the vgamt.pt file do not have a model key,
but are specific to each layer and have a 'module' prefix.
If you use released model,
you need to reorganize the original vgamt_{fr,de,cs}.pt in order to make inference.

"""


import os
import torch


def reorganize_to_model_format(checkpoint_path, output_path):
    """
    Reorganizes the checkpoint to include a 'model' key containing all the model weights.

    Args:
        checkpoint_path (str): The path to the input checkpoint file.
        output_path (str): The path where the reorganized checkpoint will be saved.
    """
    if not os.path.isfile(checkpoint_path):
        print(f"Error: The checkpoint file {checkpoint_path} does not exist.")
        return

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Initialize the reorganized checkpoint dictionary
    reorganized_checkpoint = {}

    # Create a 'model' key to store the model weights
    reorganized_checkpoint['model'] = {}

    for key, value in checkpoint.items():
        if key.startswith('module.model'):
            # Remove the 'module.' prefix
            new_key = key[len('module.'):]
            reorganized_checkpoint['model'][new_key] = value
        else:
            reorganized_checkpoint[key] = value

    # Save
    torch.save(reorganized_checkpoint, output_path)
    print(f"Reorganized checkpoint saved to {output_path}")

if __name__ == "__main__":
    checkpoint_path = '/PATH/TO/ORIGINAL CHECKPOINT'
    output_path = '/PATH/TO/OUTPUT'
    reorganize_to_model_format(checkpoint_path, output_path)
