import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout
from tensorflow.keras.models import Sequential
print("Périphériques physiques disponibles :", tf.config.list_physical_devices())

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

full_validation_set = tf.keras.utils.image_dataset_from_directory(
   'valid',
   labels="inferred",
   label_mode="categorical",
   class_names=None,
   color_mode="rgb",
   batch_size=32, #Taille des batchs (number of samples - rows of data)
   image_size=(224, 224),
   shuffle=False,
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

#Apprentissage par transfert (Transfer learning)
model = tf.keras.models.load_model("trained_model_efficient1.keras") #Changer nom si rerun
base_model = model.layers[1] #Bloc EfficientNet - Index 1 structure sequential

model.summary()

# ------------------------------------------------------------
# for i, layer in enumerate(model.layers): #Layers of the model to find EfficientNet index
#     print(i, layer.name, type(layer))

# for i, layer in enumerate(base_model.layers): #Layers of the base model (EfficientNet)
#     print(i, layer.name)
#------------------------------------------------------------

# 2 FINE-TUNING: technique d'apprentissage par transfert consistant à prendre un modèle pré-entraîné sur un jeu de données géant (base_model), puis à réentraîner une partie de ses couches sur un nouveau jeu de données plus spécifique.

base_model.trainable = True

# Geler toutes les couches
for layer in base_model.layers:
   layer.trainable = False

# Dégeler les 20 dernières couches (sortie)
for layer in base_model.layers[-20:]:
   if not isinstance(layer, tf.keras.layers.BatchNormalization): # Si il ne s'agit pas d'une couche de BatchNormalization,
      layer.trainable = True # Entraîner la couche.
   else:
      layer.trainable = False # Aussi non, aucun entraînement des couches en question (pour éviter la baisse d'accuracy).
   print(layer.name, layer.trainable)

print("Total des couches entraînables :", sum([layer.trainable for layer in base_model.layers]))

# Re-compilation avec un Learning Rate très faible pour le fine-tuning (micro-ajustements modèle)
model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.00001),loss='categorical_crossentropy',metrics=['accuracy'])

# Model Training - Adaptation 
training_history = model.fit(x=training_set,validation_data=validation_set,epochs=5)

# Model Evaluation on Training set
train_loss, train_acc = model.evaluate(training_set)

print(train_loss, train_acc)

#Model on Validation set
val_loss, val_acc = model.evaluate(validation_set)

print(val_loss, val_acc)

#Saving model
model.save("trained_model_efficient_finetuned1.keras") # Change name if retraining
training_history.history

#Recording History in json
import json
with open("trained_model_efficient_finetuned1.json", "w") as f:
    json.dump(training_history.history, f)

training_history.history['val_accuracy']

epochs = [i for i in range (1,6)]
plt.plot(epochs, training_history.history['accuracy'], color='red', label='Training Accuracy')
plt.plot(epochs, training_history.history['val_accuracy'], color='blue', label='Validation Accuracy')
plt.xlabel("No. of Epochs")
plt.ylabel("Accuracy Result")
plt.title("Visualization of Accuracy Result - Fine-Tuning")
plt.legend()
plt.show()

# 3. Evaluation of final metrics for model evaluation (Test set virtuel 10%) 

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

print("\n=== CLASSIFICATION REPORT FINE-TUNED (SUR LE TEST SET - 10%) ===")
print(classification_report(Y_true, predicted_categories, target_names=class_name))

cm = confusion_matrix(Y_true, predicted_categories)
cm.shape

# Confusion Matrix Visualization
plt.figure(figsize=(40,40))
sns.heatmap(cm, annot=True, annot_kws={'size': 10})
plt.xlabel("Predicted Class", fontsize=20)
plt.ylabel("Actual Class", fontsize=20)
plt.title("Plant Disease Prediction Confusion Matrix Fine-Tuned (Final)", fontsize=25)
plt.show()

# 3. CRÉATION D'UN TROISIÈME FICHIER (EXPORTER, robustesse + ECE + T scaling)
