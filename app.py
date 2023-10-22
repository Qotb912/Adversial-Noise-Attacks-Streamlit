import streamlit as st
import os
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import load_model

# Helper function to preprocess the image so that it can be inputted in MobileNetV2
def preprocess(image):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (224, 224))
  image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
  image = image[None, ...]
  return image
def preprocess_org(image):
  image = tf.cast(image, tf.float32)
  image = image[None, ...]
  return image
# Helper function to extract labels from probability vector
def get_imagenet_label(probs):
  return decode_predictions(probs, top=1)[0][0]
def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = pretrained_model(input_image)
    loss = loss_object(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad
def display_images(image, description):
  _, label, confidence = get_imagenet_label(pretrained_model.predict(image))
  plt.figure()
  plt.imshow(image[0]*0.5+0.5)
  plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
                                                   label, confidence*100))
  plt.show()

def save_uploadedfile(uploadedfile):
     with open(os.path.join("./",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success(f"{uploadedfile.name} File Uploaded Successfully!")

@st.experimental_singleton
def load_attack_model(suppress_st_warning=True):
    pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
    #pretrained_model = tf.keras.models.load_model('mobilenet_v2.h5')
    pretrained_model.trainable = False
    return pretrained_model
################################################################################
# Add a title 
st.markdown("<h1 style='text-align: center; color: red;'> Adversial Noise Attacks </h1>", unsafe_allow_html=True)

pretrained_model = load_attack_model()

image_file = st.file_uploader('Upload image file')
col1, col2 = st.columns(2)
if image_file:
    #save_uploadedfile(image_file)
    bytes_data = image_file.getvalue()
    col1.image(bytes_data, caption='Original Image')

    with st.spinner('Working on it...'):
        # ImageNet labels
        decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

        #image_raw = tf.io.read_file(image_file.name)
        image = tf.image.decode_image(bytes_data)

        image_original = preprocess_org(image)
        image = preprocess(image)
        image_probs = pretrained_model.predict(image)

        _, image_class, class_confidence = get_imagenet_label(image_probs)
        #print(image_class,class_confidence*100)
        col1.write(image_class)
        col1.write(class_confidence*100)
        #plt.figure()
        #plt.imshow(image[0] * 0.5 + 0.5)  # To change [-1, 1] to [0,1]
        #plt.title('{} : {:.2f}% Confidence'.format(image_class, class_confidence*100))
        #plt.show()

        loss_object = tf.keras.losses.CategoricalCrossentropy()

        # Get the input label of the image.
        labrador_retriever_index = 208
        label = tf.one_hot(labrador_retriever_index, image_probs.shape[-1])
        label = tf.reshape(label, (1, image_probs.shape[-1]))

        perturbations = create_adversarial_pattern(image, label)
        #plt.imshow(perturbations[0] * 0.5 + 0.5);  # To change [-1, 1] to [0,1]


        epsilons = [0.07]
        eps = 0.07
        #descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
        #                for eps in epsilons]

        #for i, eps in enumerate(epsilons):
        #    adv_x = image + eps*perturbations
        #    adv_x = tf.clip_by_value(adv_x, -1, 1)
        #    #display_images(adv_x, descriptions[i])
            
        im = image_original + tf.image.resize((eps*perturbations), (image_original.shape[1], image_original.shape[2]))

        image_probs = pretrained_model.predict(im)
        _, image_class, class_confidence = get_imagenet_label(image_probs)
        #print(image_class,class_confidence*100)
        

        tf.keras.preprocessing.image.save_img(f'processed_{image_file.name}.png',im[0]*0.5+0.5)    
        col2.image(f'processed_{image_file.name}.png',caption='Processed Image')
        col2.write(image_class)
        col2.write(class_confidence*100)

        #os.remove(image_file.name)
        os.remove(f'processed_{image_file.name}.png')
    st.success('Done!') 

#########################################################
st.markdown("<h1 style='font-size:20px; text-align: center; color: green; font-family:SansSerif;'>Made with 💖 By Ahmed Hossam</h1>", unsafe_allow_html=True)
st.markdown("[My Github](https://github.com/Ahmed-Hossam-Aldeen)")


st.markdown("<h1 style='font-size:40px; text-align: left; color: red; font-family:SansSerif;'>Free<br>Palestine </h1>", unsafe_allow_html=True)
st.image('https://upload.wikimedia.org/wikipedia/commons/0/00/Flag_of_Palestine.svg', width=50)
