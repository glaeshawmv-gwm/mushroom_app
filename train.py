# Install compatible versions
!pip install tensorflow==2.18.1 scikit-learn==1.3.0 imbalanced-learn==0.11.1 matplotlib opencv-python
# Imports
import numpy as np
import pandas as pd
import os
import cv2
import shutil
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as efficientnet_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
print("All imports successful.")


BASE_DIR = r"C:\Users\Glaesha\Documents\Thesis\DMV Thesis App\mushroom_app_flask\mushroom_dataset"
CLASSES = [
    "contamination_bacterialblotch",
    "contamination_cobweb",
    "contamination_greenmold",
    "healthy_bag",
    "healthy_mushroom"
]
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
RANDOM_STATE = 42


data = []
for cls in CLASSES:
    folder = os.path.join(BASE_DIR, cls)
    for file in os.listdir(folder):
        if file.lower().endswith(("jpg","jpeg","png")):
            data.append([os.path.join(folder, file), cls])
df = pd.DataFrame(data, columns=["path", "label"])
print(df.head(), "\nTotal images:", len(df))


train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=RANDOM_STATE)
val_df, test_df   = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=RANDOM_STATE)
print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))


train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.10,
    zoom_range=0.20,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=efficientnet_preprocess
)
val_datagen = ImageDataGenerator(preprocessing_function=efficientnet_preprocess)
test_datagen = ImageDataGenerator(preprocessing_function=efficientnet_preprocess)
train_gen = train_datagen.flow_from_dataframe(
    train_df, x_col="path", y_col="label",
    class_mode="categorical",
    classes=CLASSES,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_gen = val_datagen.flow_from_dataframe(
    val_df, x_col="path", y_col="label",
    class_mode="categorical",
    classes=CLASSES,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)
test_gen = test_datagen.flow_from_dataframe(
    test_df, x_col="path", y_col="label",
    class_mode="categorical",
    classes=CLASSES,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224,224,3))
base_model.trainable = False  # Freeze to keep Random Forest workflow
inp = tf.keras.Input(shape=(224,224,3))
x = base_model(inp, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
feature_extractor = tf.keras.Model(inputs=inp, outputs=x)
print("Feature extractor ready.")


def extract_all(gen, extractor):
    feats, labels = [], []
    for x, y in gen:
        f = extractor.predict(x, verbose=0)
        feats.append(f)
        labels.append(np.argmax(y, axis=1))
        if len(feats) * gen.batch_size >= gen.n:
            break
    return np.vstack(feats), np.hstack(labels)
X_train, y_train = extract_all(train_gen, feature_extractor)
X_val, y_val     = extract_all(val_gen, feature_extractor)
X_test, y_test   = extract_all(test_gen, feature_extractor)
print("Train features shape:", X_train.shape)


rf = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train, y_train)


cal_rf = CalibratedClassifierCV(rf, method='sigmoid', cv='prefit')
cal_rf.fit(X_val, y_val)


val_probs = cal_rf.predict_proba(X_val)
val_maxp = np.max(val_probs, axis=1)
# Threshold: mean - 2*std
THRESHOLD = np.mean(val_maxp) - 2*np.std(val_maxp)
print("Robust threshold for not_mushroom:", THRESHOLD)


mean_vec = np.mean(X_train, axis=0)
cov_mat  = np.cov(X_train.T)
inv_cov  = np.linalg.pinv(cov_mat)  # pseudo-inverse for stability
print("Mahalanobis distance ready.")


def predict_image_open_set_safe(img_path, extractor, model, classes, threshold=THRESHOLD, tta_steps=8, mahal_thresh=3.5):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError("Image not found.")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    preds, feats = [], []
    for _ in range(tta_steps):
        img = cv2.resize(img_rgb, IMG_SIZE)
        x = efficientnet_preprocess(img.astype(np.float32))
        x = np.expand_dims(x, axis=0)
        feat = extractor.predict(x, verbose=0)
        feats.append(feat[0])
        pred = model.predict_proba(feat)[0]
        preds.append(pred)
    avg_pred = np.mean(preds, axis=0)
    max_prob = np.max(avg_pred)
    pred_idx = np.argmax(avg_pred)
    avg_feat = np.mean(feats, axis=0)
    m_dist = mahalanobis(avg_feat, mean_vec, inv_cov)
    # Safer detection logic
    if max_prob < threshold:
        return "not_mushroom", max_prob, m_dist, avg_pred
    elif m_dist > mahal_thresh and max_prob < 0.7:
        return "not_mushroom", max_prob, m_dist, avg_pred
    else:
        return classes[pred_idx], max_prob, m_dist, avg_pred


img_path = r"C:\Users\Glaesha\Downloads\7kdzbdoapka71.jpg"
pred, conf, m_dist, raw_probs = predict_image_open_set(
    img_path, feature_extractor, cal_rf, CLASSES, THRESHOLD, tta_steps=8, mahal_thresh=3.0
)
print("Prediction:", pred)
print("Confidence (RF max prob):", conf)
print("Mahalanobis distance:", m_dist)


img_path = r"C:\Users\Glaesha\Downloads\66e5f885366578c56458825f.jpg"
pred, conf, m_dist, raw_probs = predict_image_open_set(
    img_path, feature_extractor, cal_rf, CLASSES, THRESHOLD, tta_steps=8, mahal_thresh=3.0
)
print("Prediction:", pred)
print("Confidence (RF max prob):", conf)
print("Mahalanobis distance:", m_dist)


# Tree (unrelated)
img_path = r"C:\Users\Glaesha\Downloads\06e51885665578356458825f.jpg"
pred, conf, m_dist, _ = predict_image_open_set_safe(img_path, feature_extractor, cal_rf, CLASSES)
print(pred, conf, m_dist)
# Green mold (contamination)
img_path = r"C:\Users\Glaesha\Downloads\Screenshot 2025-08-05 113550.jpg"
pred, conf, m_dist, _ = predict_image_open_set_safe(img_path, feature_extractor, cal_rf, CLASSES)
print(pred, conf, m_dist)


