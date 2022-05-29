VGG=keras.applications.vgg19.VGG19(input_shape=(224,224,3), include_top=False, weights='imagenet')
VGG.trainable=False

drive.mount('/content/drive', force_remount=True)
os.chdir('/content/drive/MyDrive/images')

os.chdir('/content/drive/MyDrive/images')
data_path = os.getcwd()

train_path = os.path.join(data_path, 'training')
test_path = os.path.join(data_path, 'validation', 'gallery')

batch_size = 32

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_path,
    labels='inferred',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(224, 224),
    shuffle=True,
    seed=42,
    subset='training',
    validation_split=0.2)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    train_path,
    labels='inferred',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(224, 224),
    shuffle=False,
    seed=42,
    subset='validation',
    validation_split=0.2)


inputs = tf.keras.Input((224, 224, 3))
x = VGG(inputs, training=False)
x = layers.Flatten(name='flatten_39')(x)
x = layers.Dense(units=124, activation='softmax', kernel_regularizer=regularizers.L1L2())(x)

model = tf.keras.Model(inputs, x)



model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])

history = model.fit(train_dataset, validation_data=test_dataset, epochs=4)
