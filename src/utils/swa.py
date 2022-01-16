import os
import torch


def swa(my_model, saved_model_folder, saved_model_list):
    print(saved_model_folder)
    saved_models = []
    for i in saved_model_list:
        f_n = os.path.join(saved_model_folder, "params_epoch_{}.ckpt".format(i))
        if os.path.exists(f_n):
            saved_models.append(torch.load(f_n))
    saved_models_num = len(saved_models)
    model_keys = saved_models[-1].keys()
    state_dict = saved_models[-1]
    new_state_dict = state_dict.copy()
    for key in model_keys:
        sum_weight = 0.0
        for m in saved_models:
            sum_weight += m[key]
        avg_weight = sum_weight / saved_models_num
        new_state_dict[key] = avg_weight
    print("averaging {} models' parameters".format(saved_models_num))
    my_model.load_state_dict(new_state_dict)
