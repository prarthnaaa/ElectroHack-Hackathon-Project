import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_DIR = r"C:\Users\Administrator\Desktop\vscode\ElectroHack\agri_data\images"
LABEL_DIR = r"C:\Users\Administrator\Desktop\vscode\ElectroHack\agri_data\labels"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

def yolo_to_ssd(yolo_box, img_width, img_height):
    x_center, y_center, width, height = yolo_box
    x_min = int((x_center - width / 2) * img_width)
    y_min = int((y_center - height / 2) * img_height)
    x_max = int((x_center + width / 2) * img_width)
    y_max = int((y_center + height / 2) * img_height)
    return [x_min, y_min, x_max, y_max]

def check_labels(label_dir):
    class_counts = {}
    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            with open(os.path.join(label_dir, file), 'r') as f:
                for line in f:
                    class_id = line.strip().split()[0]
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
    print("Class Distribution in Labels:", class_counts)
    return class_counts

def load_and_preprocess_images(image_dir, label_dir):
    images, labels = [], []
    print(f"Loading images from: {image_dir}")
    print(f"Loading labels from: {label_dir}")

    if not os.path.exists(image_dir):
        print(f"Error: Image directory does not exist: {image_dir}")
        return np.array([]), np.array([])

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} image files")

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, img_file.replace('.jpg', '.txt').replace('.jpeg', '.txt'))

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        img_height, img_width = img.shape[:2]
        img = cv2.resize(img, (224, 224))  
        img = preprocess_input(img)

        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, *box = map(float, line.strip().split())
                    boxes.append(yolo_to_ssd(box, img_width, img_height))
        else:
            print(f"Label file not found: {label_path}")

        labels.append(1 if any(box[0] != 0 for box in boxes) else 0)  
        images.append(img)

    print(f"Successfully loaded {len(images)} images")
    return np.array(images), to_categorical(labels, num_classes=2)

def create_mobilenet_ssd(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

def process_and_detect_weeds():
    class_counts = check_labels(LABEL_DIR)
    if len(class_counts) < 2:
        print("⚠ Warning: Dataset may be imbalanced or incorrectly labeled.")

    X, y = load_and_preprocess_images(IMAGE_DIR, LABEL_DIR)

    if X.shape[0] == 0 or y.shape[0] == 0:
        print("Error: No images were loaded. Check dataset.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    datagen = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
    )
    
    X_train_augmented, y_train_augmented = next(datagen.flow(X_train, y_train, batch_size=len(X_train)))
    X_train = np.concatenate([X_train, X_train_augmented])
    y_train = np.concatenate([y_train, y_train_augmented])

    model = create_mobilenet_ssd(num_classes=2)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=5, batch_size=8, validation_split=0.2, verbose=1)

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"🔹 Test Accuracy: {test_accuracy*100:.2f}%")

    if X_test.shape[0] > 0:
        sample_img = X_test[0]
        predictions = model.predict(np.expand_dims(sample_img, axis=0))
        detected_class = np.argmax(predictions[0])
        print(f"Detected Class: {'Weed' if detected_class == 1 else 'No Weed'}")
    else:
        print("No test images available for detection.")

    model.save("weed_detection_model.h5")
    print("Model saved as 'weed_detection_model.h5'")

if __name__ == "__main__":
    process_and_detect_weeds()