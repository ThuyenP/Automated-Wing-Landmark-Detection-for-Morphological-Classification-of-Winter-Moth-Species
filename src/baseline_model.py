from resNet import resNet_helper
from GM import GM_helper

# Note param for each model:
    # ResNet: 2 possible resNet model name - 'ResNet18' and 'ResNet50'
    # GM: label_dir="GM labels", wing_ID = "General Data/Wing IDs - Label.csv"

# DeepWings method
def deepWings():
    pass

# ResNet method
def resNet(model_name, data_dir, save_dir, model_save_path, \
           num_classes, batch_size, num_epochs, num_workers, image_size):
    return resNet_helper(model_name, data_dir, save_dir, model_save_path, \
           num_classes, batch_size, num_epochs, num_workers, image_size)

# Classic GM
def GM(label_dir, wing_ID, output_dir):
    GM_helper(label_dir, wing_ID, output_dir)