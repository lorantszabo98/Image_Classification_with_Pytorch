import os
import torch


def save_model(save_directory, model, feature_extractor_mode=False):
    os.makedirs(save_directory, exist_ok=True)
    model_name = model.__class__.__name__
    if feature_extractor_mode:
        torch.save(model.state_dict(), os.path.join(save_directory, f"{model_name}_model_feature_extractor.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_directory, f"{model_name}_model_fine_tuned.pth"))


def load_model(load_directory, model, feature_extractor_mode=False):
    model_name = model.__class__.__name__
    if feature_extractor_mode:
        model.load_state_dict(torch.load(os.path.join(load_directory, f"{model_name}_model_feature_extractor.pth")))
    else:
        model.load_state_dict(torch.load(os.path.join(load_directory, f"{model_name}_model.pth")))