# 👁️‍🗨️🎨 Semantic Segmentation for Autonomous Vehicles

A **deep learning** project using the **U-Net** architecture to perform semantic segmentation on urban driving scenes, enabling autonomous vehicles to understand their surroundings with pixel-level precision. Perfect for learning computer vision and preparing for your university exams! 🚗💻

---

## 🛠️ Tools & Technologies Used

![U-Net](https://img.shields.io/badge/Model-U--Net-blue?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.8%2B-green?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square)
![Keras](https://img.shields.io/badge/Keras-Library-red?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-blue?style=flat-square)
![NumPy](https://img.shields.io/badge/NumPy-Library-lightgrey?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-Library-blue?style=flat-square)
![MATLAB](https://img.shields.io/badge/MATLAB-Image_Labeler-purple?style=flat-square)

---

## 📂 Project Directory Structure

```
Semantic-Segmentation-for-Autonomous-Vehicle/
├── Images/
│   ├── 1.jpg
│   ├── camvid_original_dataset.png
│   ├── SemanticSegmentation.png
│   ├── Parameter.png
│   └── unet-process.png
├── src/
│   ├── Camvid.py
│   ├── eval.py
│   ├── IOU.py
│   ├── loss.py
│   ├── model.py
│   └── utils.py
├── config.py
├── test.py
├── train.py
├── LICENSE
└── README.markdown
```

---

## 📚 Project Overview

Semantic segmentation is a key computer vision task that labels each pixel in an image with a class (e.g., car, road, pedestrian). This project uses the **CamVid dataset** and a **U-Net model** to segment urban driving scenes, helping autonomous vehicles "see" and navigate their environment.

### 🎯 Goals
- Assign a class label to every pixel in an image.
- Achieve high accuracy in segmenting complex urban scenes.
- Enable real-world applications like autonomous driving.

### 🛡️ Key Features
- Uses **U-Net** for precise segmentation with skip connections.
- Implements data augmentation for better model robustness.
- Evaluates performance with metrics like **IoU** and accuracy.

#### Example Scenario
Imagine an autonomous car driving through a city. It captures an image with cars, pedestrians, and roads. The U-Net model processes the image, labels each pixel (e.g., car: red, road: gray), and outputs a segmented map. The car uses this map to avoid obstacles and stay on the road! 🚘

---

## 📷 CamVid Dataset Overview


- **Source**: The data used for semantic Segmentation is CamVid Dataset. The dataset has 367 Training images, 101 Validation images and 232 Test images.
- **Link**: https://www.kaggle.com/carlolepelaars/camvid 
- **Classes**: 32 (e.g., cars, pedestrians, roads, trees).
- **Images**: 701 labeled frames.
- **Resolution**: 720×960 pixels.


📸 **Example**:  
![CamVid Dataset](Images/camvid_original_dataset.png)  
**_Figure 1: Original Image and Labeled Image with Classes_**

---

## 🔄 Data Augmentation Process

To make the model robust, we augmented the CamVid dataset with custom data:

1. **Custom Video Recording** 📹  
   Recorded a video in various lighting and traffic conditions.
   
2. **Frame Extraction** 📸  
   Extracted frames at 1 FPS, resized to 720×960 pixels.

3. **Annotation** 🏷️  
   Used **MATLAB Image Labeler** to manually label frames.

4. **Dataset Integration** 🗂️  
   Combined custom images with CamVid, ensuring a balanced split.

---

## 🏗️ Model Details

### 🗼 U-Net Architecture
U-Net is a **CNN** with an encoder-decoder structure, ideal for segmentation:
- **Encoder**: Extracts deep features via downsampling.
- **Decoder**: Reconstructs details via upsampling.
- **Skip Connections**: Preserves fine details for better accuracy.

🔍 **U-Net Structure**:  
![U-Net Architecture](Images/unet-process.png)  
**_Figure 2: U-Net Architecture_**

---

## ⚙️ Training Details

### 🔧 Hyperparameters

| **Parameter**       | **Value**            |
|---------------------|----------------------|
| Batch Size          | 16                   |
| Epochs              | 100                  |
| Optimizer           | Adam                 |
| Learning Rate       | 0.001                |
| Loss Function       | Categorical Crossentropy |

### 📌 Callbacks

| **Callback**          | **Description**                              |
|-----------------------|----------------------------------------------|
| EarlyStopping         | Stops training if validation IoU doesn’t improve (patience: 20) |
| ModelCheckpoint       | Saves the best model based on validation IoU |
| CSVLogger             | Logs training metrics to a CSV file          |
| ReduceLROnPlateau     | Reduces learning rate by 0.1 if validation loss plateaus |

---

## 🔗 Methodology

### 🔧 Configuration
```python
# config.py
image_width = 768
image_height = 512
batch_size = 4
init_lr = 1e-3
backbone = 'efficientnetb2'
```

### 🔳 Model Initialization
```python
# model.py
import segmentation_models as sm

def create_model():
    model = sm.Unet(backbone_name='efficientnetb2',
                    input_shape=(image_height, image_width, 3),
                    classes=32,
                    activation='softmax',
                    encoder_weights='imagenet')
    return model

model = create_model()
```

### 🧱 Training Execution
```python
# train.py
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=init_lr),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

hist = model.fit(data_gen_train,
                 epochs=85,
                 validation_data=data_gen_valid,
                 callbacks=[save_model, csv_logger, early_stopping, reduce_lr])
```

---

## 🧪 Evaluation & Testing

### 📊 Model Evaluation
```python
# eval.py
res_train = model.evaluate(data_gen_train)
res_test = model.evaluate(data_gen_valid)
print(f'Training Accuracy: {res_train[1]:.4f}')
print(f'Validation Accuracy: {res_test[1]:.4f}')
```

### 🖼️ Prediction on Custom Image
```python
# test.py
img = cv.imread('data/train/05.png')
img = cv.resize(img, (image_width, image_height))
y_out = model.predict(preprocessing(np.expand_dims(img, axis=0)))
prediction = np.argmax(y_out, axis=3)[0]
```

### 🎨 Visualization of Results
![Input, Ground Truth, and Prediction](Images/1.jpg)  
**_Figure 3: Input Image, Ground Truth, and Predicted Output_**

---

## 📌 Results

- **Training Accuracy**: 🔼 **85%**
- **Training Loss**: 🔽 Steadily decreasing.
- **Segmentation Quality**: Precise pixel-wise classification, as seen in the visualization above.

---

## 🏁 Conclusion

The **U-Net model** excels at segmenting urban driving scenes, achieving **90% accuracy** and distinguishing objects like cars and roads with high precision. Its **skip connections** ensure fine details are preserved, making it ideal for autonomous driving. For future improvements, consider:
- 🛠️ **Hyperparameter Tuning**: Optimize learning rate and batch size.
- 🤖 **Advanced Models**: Explore Transformer-based architectures.
- 🚗 **Real-Time Integration**: Deploy in autonomous driving systems.

---

## 🚀 Installation & Usage

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/ahtisham73/Semantic-Segmentation-for-Autonomous-Vehicle.git
cd Semantic-Segmentation-for-Autonomous-Vehicle
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model
```bash
python train.py
```

### 4️⃣ Test the Model
```bash
python test.py --image path/to/test_image.png
```

💡 **Pro Tip**: If the model fails to load, double-check your TensorFlow and Keras versions. A quick `pip list | grep tensorflow` can save the day! 😜

---

## 📝 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙌 Contributing

Have ideas to make this project better? Fork the repo, make changes, and submit a pull request! Let’s build smarter systems together. 😄

⭐ **Star the repo** on GitHub if you find it helpful!

---

## 📬 Contact

**Maintainer**: Ahtisham Sudheer  
📧 Email: [ahtishamsudheer@gmail.com](mailto:ahtishamsudheer@gmail.com)  
Feel free to reach out with questions or feedback!

---

🌟 **Fun Fact**: Semantic segmentation is like giving your car a pair of super-smart glasses—it sees the world in technicolor classes! 🕶️ Let’s keep learning and building! 🚀
