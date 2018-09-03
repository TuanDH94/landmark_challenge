# Zalo AI Challenge
# Giới thiệu
Zalo AI challenge là cuộc thi AI đầu tiên do zalo tổ chức. Nội dung liên quan đến xử dụng âm thanh, hình ảnh và các loại dữ liệu khác. 
# Data augmentation
Các phương pháp data augmentation được áp dụng bao gồm:
* Flip
* Random rotation
* Shear, zoom
* height shift
# Fine tuning
Sử dụng 8 mạng CNN có top-1 và top-5 accuarcy trên imagenet cao nhất, bao gồm:
* VGG16
* VGG19
* Inception V3
* ResNet50
* InceptionResNet
* Xception
* Desnet201
* NasNetMobile
# Ensemble kết quả
Ensemble 8 mạng CNN trên bằng phương pháp voting.
# Train
Các bạn cần tải tập train và test, rồi để vào folder như trong code, sau đó chạy lệnh sau để split tập train và valid.
python train_valid_data.python
train model
python inception_resnet_v2_training_code.py
Model được lưu với tên keras_landmark_inception_resnet_v2_model.0.01632.h5
# Test
python testing_code.py
