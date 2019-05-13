from keras.preprocessing import image

model = load_model('HorseHumanClassifier.h5')

path = '#path_to_image'
img = image.load_img(path, target_size = (300, 300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
im = np.vstack([x])

classes = model.predict(im)
print(classes[0])
if(classes[0]>0.5): print('human')
else: print('horse')
