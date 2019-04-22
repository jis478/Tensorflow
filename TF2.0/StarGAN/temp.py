
#     real = tf.ones_like(fake_D_src)
#     fake = tf.zeros_like(fake_D_src)

#     # step(a-1)
#     D_src_loss_original = cross_entropy(real,real_D_src)
#     # step(a-2)
#     D_src_loss_target = cross_entropy(fake,fake_D_src)
#     # step(a-1 + a-2)
#     D_src_loss = D_src_loss_target + D_src_loss_original

#     ### generator loss
#     # step(d)
#     G_src_loss_target = cross_entropy(real, fake_D_src)
#     G_src_loss = G_src_loss_target

#     ### Domain cls loss
#     # step (a-1)
#     D_cls_loss_target = cross_entropy(original_domain, original_D_cls)
#     # step (d)
#     G_cls_loss_target = cross_entropy(target_domain, target_D_cls)

#     ### Reconstruction loss
#     Rec_loss = np.abs(tf.reduce_mean(input_image  - reconstructed_image))
    
#     G_loss = G_src_loss + lambda_cls * G_cls_loss_target + lambda_rec * Rec_loss 
#     D_loss = -D_src_loss + lambda_cls * D_cls_loss_target
     
