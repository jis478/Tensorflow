

### Loss
def adverserial_loss(logits, real=True):
  if real == True:
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.ones_like(logits),logits = logits)
  else:
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.zeros_like(logits),logits = logits)
  return loss

def reconstruction_loss(image,rec_image):
  return lambda_rec * np.abs(tf.reduce_mean(image  - rec_image))
 
def domain_cls_loss(domain, logits):
  return lambda_cls * tf.losses.sigmoid_cross_entropy(multi_class_labels = domain, logits = logits)

## G_loss, D_loss
def G_loss(fake_D_src, target_D_cls, target_domain,input_image, reconstructed_image, lambda_cls,lambda_rec):
  loss = adverserial_loss(fake_D_src, real=True) + lambda_cls * domain_cls_loss(target_domain, target_D_cls) + lambda_rec * reconstruction_loss(input_image, reconstructed_image)
  return loss
  
def D_loss(real_D_src, fake_D_src, original_domain, original_D_cls, lambda_cls):
  loss = -1 * (adverserial_loss(real_D_src, real=True) + adverserial_loss(fake_D_src, real=False)) + lambda_cls* domain_cls_loss(original_domain,original_D_cls)
  return loss


# ## Optimizer
generator_optimizer = tf.train.AdamOptimizer(1e-4)
discriminator_optimizer = tf.train.AdamOptimizer(1e-4)


# ## Train loop
def train_step(input_image, original_domain, target_domain):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        
    # generator + discriminator combined
    fake_image = generator(input_image, target_domain)  # step(b)
    fake_D_src, target_D_cls = discriminator(fake_image)  # step(d) # 우선 fake image를 넣어서 보조 classification을 학습
    reconstructed_image = generator(fake_image, original_domain) # step(c)

    # discriminator
    real_D_src, original_D_cls = discriminator(input_image) #step(a) 
    fake_D_src, fake_D_cls = discriminator(fake_image) #step(a)  # 여기서는 보조 classification 학습 안함
    
    generator_loss = G_loss(fake_D_src, target_D_cls, target_domain,input_image, reconstructed_image, lambda_cls,lambda_rec)
    discriminator_loss = D_loss(real_D_src, fake_D_src, original_domain, original_D_cls, lambda_cls)
    
    gradients_of_generator = gen_tape.gradient(generator_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    print()

    
def train(train_dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for input_image, original_domain in train_dataset:
            
      target_domain = random_target_domain_generation()
      train_step(input_image, original_domain, target_domain)
     
