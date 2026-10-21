Step to run the code:

    1. Run COCO_converter.py to create labels 
    2. Run clahe_ing.py to transform the image
    3. Run yolo_pose_test.py to train the network
    4. Run best_pose_model_test.py to test the network 

(Run when conda deactivate)

Instruction when run on Unity:
  1. To load the data to VM, connect to Globus to transfer file
  2. Have to allocate resource to run the code. To do that, run this command: 
    srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=04:00:00 --pty bash
  3. Set patience = 50 or 100 to halt the epochs execution when mAP is not improving

Command line: 
  1. Run job: sbatch train_yolo_pose.sh
  2. Check on queue: squeue -u thuyenpham_umass_edu
  3. View output file: tail -f [file_name].out
  4. Check on idle gpu: sinfo -p gpu -o "%N %G %t" | grep idle

Note of yolo run:
  1. YOLO without class label: epochs = 200, patience = 50
  2. YOLO with class label (v1): epochs = 70, patience = 10
  3. YOLO with class label (v2): epochs = 200, patience = 50
  4. YOLO with class label (v3): epochs = 200, patience = 50
      - Only train yolo11x and yolo26x
      - Do epochs = [100, 150, 200]
  -> best one: yolo26x, epochs = 200


Explanation on the metric result: 
  1. box_loss: how accurately the model predicts the location and size of the bounding boxes 
               around objects. A lower box_loss value indicates that the predicted boxes are 
               closely aligned with the actual "ground truth" boxes in the training data. 
               It is typically calculated using loss functions such as Complete Intersection 
               over Union (CIoU) loss.
  2. pose_loss: a specific component of the total loss function that measures the error between
               predicted keypoint coordinates and ground truth keypoints. It ensures accurate 
               localization of joints or keypoints on detected objects, contributing 
               significantly to training performance alongside bounding box and classification losses.
  3. cls_loss: (Classification Loss) This evaluates the model's performance in correctly identifying 
              the class or category of the detected objects. It typically uses a form of cross-entropy 
              loss (specifically, Varifocal Loss in some YOLOv8 variants) to measure the error in the 
              predicted class probabilities compared to the true labels. A lower cls_loss means the model 
              is better at classifying objects correctly.
  4. kobj_loss: (keypoint object loss) represents the confidence loss associated with keypoint predictions. 
              It measures how accurately the model distinguishes between valid keypoints and 
              background noise.
  5. dfl_loss: (Distribution Focal Loss) refine the bounding box predictions by treating the bounding box 
              regression as a probability distribution problem. Instead of predicting fixed coordinates, 
              it helps the model focus on hard-to-classify examples and objects with ambiguous boundaries, 
              enhancing the precision and discriminative power of the final bounding boxes.
  
Implementation on UNet: https://www.geeksforgeeks.org/machine-learning/u-net-architecture-explained/ 