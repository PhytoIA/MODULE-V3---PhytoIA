#Importations (libraries)
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from sklearn.metrics import classification_report, confusion_matrix
print(tf.config.list_physical_devices())

#1. Chargement et répartition des datasets ----------------------------------

#Chargement du bloc complet de training (80% du total global) 
training_set = tf.keras.utils.image_dataset_from_directory(
   'train',
   labels="inferred",
   label_mode="categorical", #Classes (38)
   class_names=None,
   color_mode="rgb",
   batch_size=32,
   image_size=(224, 224),
   shuffle=True,
   seed=None,
   validation_split=None,
   subset=None,
   interpolation="bilinear",
   follow_links=False,
   crop_to_aspect_ratio=False,
)

#Chargement du bloc complet de validation (20% du total global) 
full_validation_set = tf.keras.utils.image_dataset_from_directory(
   'valid',
   labels="inferred",
   label_mode="categorical",
   class_names=None,
   color_mode="rgb",
   batch_size=32, #Taille des batchs (number of samples - rows of data)
   image_size=(224, 224),
   shuffle=True,
   seed=42, # Séparation étanche (de même ordre) validation-test subsets(take/skip)
   interpolation="bilinear",
   follow_links=False,
   crop_to_aspect_ratio=False,
)

# Calcul du nombre total (cardinality - number of elements) de batchs (pour la répartition)
total_batches = tf.data.experimental.cardinality(full_validation_set).numpy()
val_batches = total_batches // 2 # Diviser le nombre de batches en deux (10% validation set)

# Création des deux subsets
# - Ensemble validation pendant model.fit et optimisation du temperature scaling (T)
validation_set = full_validation_set.take(val_batches) # Number of elements of the dataset taken
# - Ensemble test pour l'ECE et l'évaluation de la robustesse (selon le protocole benchmark ImageNet-C appliqué au plantvillage dataset)
test_set = full_validation_set.skip(val_batches) # Number of elements of this dataset that should be skipped to form the second dataset 

# 2. Building Model ----------------------------------
from tensorflow.keras import layers

# Initialize the base model without final classification
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False, # Enlever classificateur final
    weights="imagenet", # Pre-training on ImageNet
    input_shape=(224,224,3) # Image layout and color (RGB)
)

# Freeze the base model to preserve learned features during intial training
base_model.trainable = False

# Data augmentation - Impliqué dans framework d'entraînement
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.2),
])

# Building model (Sequential)
model = tf.keras.Sequential([
    data_augmentation,

    base_model,

    #Flatten spatial dimensions - Average pooling - Vecteur de 1280 valeurs
    layers.GlobalAveragePooling2D(),

    layers.Dense(256, activation="relu"), # Neurones

    #To avoid overfitting (reduce)
    layers.Dropout(0.4),
    
    #Dense with number of classes 
    layers.Dense(38, activation="softmax") # Probabilités (plus grande - exponentielle)
])

#Output Layer
model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

model.build((None,224,224,3))
# EfficientNet = extraction de caractéristiques générales
# Dense(256) = adaptation aux maladies végétales
# Dense(38) = décision finale parmi tes 38 classes
# ReLU = permet au réseau d'apprendre des relations complexes
# build() = seulement préparer la structure, pas apprendre

model.summary()

# Model Training (validation set)
training_history = model.fit(x=training_set,validation_data=validation_set,epochs=10)

# Model Evaluation on Training set
train_loss, train_acc = model.evaluate(training_set)

print(train_loss, train_acc)

#Model on Validation set
val_loss, val_acc = model.evaluate(validation_set)

print(val_loss, val_acc)

#Saving model
model.save("trained_model_efficient1.keras") # Change name if retraining
training_history.history

#Recording History in json
import json
with open("training_hist_efficient.json", "w") as f:
    json.dump(training_history.history, f)

training_history.history['val_accuracy']

epochs = [i for i in range (1,11)]
plt.plot(epochs, training_history.history['accuracy'], color='red', label='Training Accuracy')
plt.plot(epochs, training_history.history['val_accuracy'], color='blue', label='Validation Accuracy')
plt.xlabel("No. of Epochs")
plt.ylabel("Accuracy Result")
plt.title("Visualization of Accuracy Result")
plt.legend()
plt.show()

# 3. Evaluation of final metrics for model evaluation (Test set virtuel 10%) ----------------------------------

#Récupération des noms des classes 
class_name = full_validation_set.class_names

# Extraction des étiquettes réelles (y) du testset
true_categories = tf.concat([y for x,y in test_set], axis=0) # Matrice empilée haut-bas (colonnes)
Y_true = tf.argmax(true_categories, axis=1) # Returns the index with the largest value across axes of a tensor (gauche-droite vecteurs)

# Prédictions sur le testset 
y_pred = model.predict(test_set)
y_pred,y_pred.shape
predicted_categories = tf.argmax(y_pred,axis=1)

from sklearn.metrics import classification_report, confusion_matrix

print("\n=== CLASSIFICATION REPORT (SUR LE TEST SET - 10%) ===")
print(classification_report(Y_true, predicted_categories, target_names=class_name))

cm = confusion_matrix(Y_true, predicted_categories)
cm.shape

# Confusion Matrix Visualization
plt.figure(figsize=(40,40))
sns.heatmap(cm, annot=True, annot_kws={'size': 10})
plt.xlabel("Predicted Class", fontsize=20)
plt.ylabel("Actual Class", fontsize=20)
plt.title("Plant Disease Prediction Confusion Matrix", fontsize=25)
plt.show()