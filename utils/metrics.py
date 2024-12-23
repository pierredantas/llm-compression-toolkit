def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def calculate_model_size(model):

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

#    buffer_size = 0
#    for buffer in model.buffers():
#        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = param_size # + buffer_size
    return size_all_mb

def calculate_reduction_rate_mb(before_model, after_model):
    reduction_rate_mb = ((calculate_model_size(before_model) - calculate_model_size(after_model)) / calculate_model_size(before_model)) * 100
    return reduction_rate_mb

def calculate_reduction_rate_param(before_model, after_model):
    param_reduction_rate = ((count_parameters(before_model) - count_parameters(after_model)) / count_parameters(before_model)) * 100
    return param_reduction_rate

def calculate_reduction_times_mb(before_model, after_model):
    reduction_times_mb = calculate_model_size(before_model)/ calculate_model_size(after_model)
    return reduction_times_mb

def calculate_reduction_times_param(before_model, after_model):
    reduction_times_param = count_parameters(before_model)/ count_parameters(after_model)
    return reduction_times_param