# Watermark Embedding Using DWT, DCT, and Arnold Transform
This project implements the algorithm proposed in the paper:

"A Novel Hybrid DCT and DWT Based Robust Watermarking Algorithm for Color Images"
by Ahmed Khaleel Abdulrahman and Serkan Ozturk.

The authors proposed a robust color image watermarking method that combines:

* Discrete Cosine Transform (DCT)

* Discrete Wavelet Transform (DWT)

* Arnold Transform for scrambling

* RGB channel separation and embedding

## Method Summary
### Cover Image Preparation:
* The input cover image is separated into its Red, Green, and Blue components.
*  Each channel undergoes a block-based 2D DCT.
*  The DCT-transformed channel is then processed using 2D Haar DWT to obtain four frequency bands (LL, LH, HL, HH).
#### sample cover image
![cover](image.png)

### Watermark Preparation:
* A grayscale watermark is first scrambled using the Arnold transform (to increase security).
* The scrambled watermark is binarized and then transformed using block-based DCT.
* The DCT coefficients of the watermark are divided into four parts.
#### watermark or Payload
![payload](image-1.png)

### Embedding:
* The watermark sub-parts are embedded additively into the four DWT bands (LL, LH, HL, HH) of the Red channel of the cover image.
(Note: The Green and Blue channels are not used for embedding in the current implementation, but can be extended.)
#### Watermarked image with strength 0.01 with PSNR: 44.19 dB and SSIM: 0.9885
![watermarked001](image-2.png)


### Reconstruction:
* The watermarked DWT bands are inverse-transformed (IDWT), and then inverse DCT (IDCT) is applied to reconstruct the watermarked Red channel.
* The final watermarked image is obtained by merging the original Blue and Green channels with the watermarked Red channel.


