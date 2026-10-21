from baseline_model import deepWings, resNet, GM

if __name__ == '__main__':
    # # Run GM
    # GM_label_dir = "GM labels"
    # GM_wing_ID = "General Data/Wing IDs - Label.csv"
    # GM_output_dir = "runs/baseline/GM"

    # print("Run GM")
    # acc, precision, recall, f1 = GM(GM_label_dir, GM_wing_ID, GM_output_dir)
    # print(f'Accuracy:{acc}, Precision: {precision}, Recall: {recall}, f1: {f1}')

    # Run ResNet
    model_name = "ResNet18"
    data_dir = "ResNet labels"
    save_dir = "runs/baseline/ResNet18_v1"
    model_save_path = "runs/baseline/ResNet18_v1/best_model.pt"
    num_classes = 2
    batch_size = 8
    num_epochs = 5
    num_workers = 0
    image_size = 224
    
    print("Run ResNet18")
    resNet(model_name, data_dir, save_dir, model_save_path, \
          num_classes, batch_size, num_epochs, num_workers, image_size)
