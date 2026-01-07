# U net with Keras for image segmentation
import numpy as np 
import os 
import cv2
from tensorflow import  keras 
from keras.layers import Layer
from sklearn.model_selection import train_test_split
# load dataset
def load_data(root, img_size=(128,128)):
    images = []
    masks= []
    for tile in sorted(os.listdir(root)):
        img_dir=os.path.join(root, tile, "images")
        mask_dir=os.path.join(root, tile, "masks")
        if not os.path.exists(img_dir):
            continue
        for f in os.listdir(img_dir):
            if not f .lower().endswith(( '.jpg')):
              continue
            img_path=os.path.join(img_dir, f) # path to image
            mask_path=os.path.join(mask_dir, os.path.splitext(f)[0]+".png")# path to mask
            if not os.path.exists(mask_path):
                continue
            img=cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)# read image 
            img=cv2.resize(img, img_size)/255.0# resize image
            mask=cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)# read mask
            mask=cv2.resize(mask, img_size)/255.0# resize mask
            mask=np.expand_dims(mask, axis=-1)# add channel dimension
            images.append(img)
            masks.append(mask)
    return np.array(images,dtype="float32"), np.array(masks, dtype="float32")
X, Y = load_data("/Users/berkebarantozkoparan/Desktop/project 7/Semantic segmentation dataset/")
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
# U-Net model
def unet_model(input_size=(128, 128, 3)):
    inputs = keras.Input(input_size)
    # Encoder
    c1 = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    p4 = keras.layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p4)
    c5 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = keras.layers.concatenate([u6, c4])
    c6 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = keras.layers.concatenate([u7, c3])
    c7 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    u8 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = keras.layers.concatenate([u8, c2])
    c8 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u8)
    c8 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c8)

    u9 = keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = keras.layers.concatenate([u9, c1])
    c9 = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u9)
    c9 = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c9)

    outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model  
# Compile and train model
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
callbacks=[
    keras.callbacks.ModelCheckpoint("unet_model.h5", save_best_only=True),
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3)
]
history=model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=20, batch_size=16, callbacks=callbacks)  
# Save the trained model
model.save("unet_final_model.h5") 
  
import matplotlib.pyplot as plt
import numpy as np
import os

def save_predictions_grid(model, X, Y, out_dir="results", n=6, thr=0.5):
    os.makedirs(out_dir, exist_ok=True)

    idxs = np.random.choice(len(X), size=min(n, len(X)), replace=False)
    preds = model.predict(X[idxs], verbose=0)
    preds_bin = (preds > thr).astype(np.float32)

    for k, i in enumerate(idxs):
        img = X[i]
        gt  = Y[i].squeeze()
        pr  = preds_bin[k].squeeze()

        # overlay (kırmızı maske bindirme)
        overlay = img.copy()
        overlay[..., 0] = np.clip(overlay[..., 0] + 0.6 * pr, 0, 1)  # red channel

        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        axes[0].imshow(img);      axes[0].set_title("Image")
        axes[1].imshow(gt, cmap="gray"); axes[1].set_title("Ground Truth")
        axes[2].imshow(pr, cmap="gray"); axes[2].set_title("Prediction")
        axes[3].imshow(overlay);  axes[3].set_title("Overlay")

        for ax in axes: ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"sample_{k}.png"), dpi=200)
        plt.close()

    print(f"Saved {len(idxs)} samples to: {out_dir}/")

# Eğitim sonrası:
save_predictions_grid(model, X_val, Y_val, out_dir="results", n=6)
