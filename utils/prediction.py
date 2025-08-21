import numpy as np

def predict_disease(model, image_array, class_names):
    preds = model.predict(image_array)
    idx = np.argmax(preds[0])
    confidence = round(100 * np.max(preds[0]), 2)
    return class_names[idx], confidence, preds[0]
