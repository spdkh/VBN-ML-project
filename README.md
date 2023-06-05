# Image reconstruction for structured illumination microscopy using an improved caGAN

gpu remote connection running code:

activate the environment: 

    conda activate tf_gpu

train with 100 epochs:

    python -m train --epoch 100

To enable unrolling with 2 unrolling layers you can use:

    python -m train --dnn_type UCAGAN --unrolling_iter 3#   D L _ p r o j e c t _ t e m p l a t e  
 #   V B N - M L - p r o j e c t  
 