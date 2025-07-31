import cv2
import numpy as np
import pywt
from scipy.fftpack import dct, idct
from watermark import block_dct2d, block_idct2d

#adapting and reusing functions from the embedded script
'''
def inverse_arnold_transform(image,original_shape,iterations=1):
    if image.shape[0] != image.shape[1]:
        raise ValueError("Arnold transform requires a square image")
    n = image.shape[0]
    transformed_image = np.zeros_like(image)
    #invert the arnold transform
    inverse_matrix = np.array([[2, -1], [-1, 1]]) # For (x_prime, y_prime) = (2x+y, x+y)
    for _ in range(iterations):
        for x_prime in range(n):
            for y_prime in range(n):
                original_x = (2*x_prime - y_prime) % n
                original_y = (-1*x_prime + y_prime) % n
                transformed_image[original_x, original_y] = image[x_prime, y_prime]
        image = transformed_image.copy()
    return cv2.resize(image, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_AREA)
'''

def inverse_arnold_transform(image, original_shape, iterations=1):
    if image.shape[0] != image.shape[1]:
        raise ValueError("Image must be square for the inverse Arnold transform.")

    n = image.shape[0]
    unscrambled_image = np.copy(image)

    for _ in range(iterations):
        # Create an empty canvas for the original image
        original_image = np.zeros_like(unscrambled_image)
        # Iterate over each pixel of the scrambled image
        for x_prime in range(n):
            for y_prime in range(n):
                # Apply the inverse Arnold's Cat Map formula
                x = (x_prime - y_prime) % n
                y = (-x_prime + 2 * y_prime) % n
                original_image[x, y] = unscrambled_image[x_prime, y_prime]
        unscrambled_image = original_image
        
    return cv2.resize(unscrambled_image, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_AREA)

'''
def inverse_arnold_transform(image, original_shape, iterations=1):
    if image.shape[0] != image.shape[1]:
        raise ValueError("Arnold Transform requires a square image.")
    
    n = image.shape[0]
    transformed = image.copy()

    for _ in range(iterations):
        temp = np.zeros_like(transformed)
        for x_prime in range(n):
            for y_prime in range(n):
                x = (2 * x_prime - y_prime) % n
                y = (-1 * x_prime + y_prime) % n
                temp[x, y] = transformed[x_prime, y_prime]
        transformed = temp

    return cv2.resize(transformed, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_AREA)
'''
'''
def convert_to_grayscale(binary_image):
    return (binary_image * 255).astype(np.uint8)
'''

def convert_binary_to_grayscale(binary_image):
    # Create a copy to avoid modifying the original array
    grayscale_image = binary_image.copy()

    # Scale the values from 0/1 to 0/255
    # Multiplying by 255 converts all 1s to 255s. The 0s remain 0.
    grayscale_image = grayscale_image * 255

    return grayscale_image

#watermark extraction process

def watermark_extraction_process(original_cover_path, watermarked_image_path, scaling_factor, arnold_iterations=1, original_watermark_shape=(512, 512)):
    #load images
    I_O = cv2.imread(original_cover_path)
    if I_O is None:
        raise FileNotFoundError(f"Original cover image not found: {original_cover_path}")
    I_W = cv2.imread(watermarked_image_path)
    if I_W is None:
        raise FileNotFoundError(f"Watermarked image not found: {watermarked_image_path}")

    block_size = 8
    #separate I_O and I_W into R, G, B components
    B_O, G_O, R_O = cv2.split(I_O)
    B_W, G_W, R_W = cv2.split(I_W)

    #perform DCT on each colour component
    R_O_DCT = block_dct2d(R_O.astype(np.float32), block_size)
    G_O_DCT = block_dct2d(G_O.astype(np.float32), block_size)
    B_O_DCT = block_dct2d(B_O.astype(np.float32), block_size)

    R_W_DCT = block_dct2d(R_W.astype(np.float32), block_size)
    G_W_DCT = block_dct2d(G_W.astype(np.float32), block_size)
    B_W_DCT = block_dct2d(B_W.astype(np.float32), block_size)

    #perfrom DWT to the DCT tranformed color components
    coeffs_R_O_DCT_DWT = pywt.dwt2(R_O_DCT, 'haar')
    coeffs_G_O_DCT_DWT = pywt.dwt2(G_O_DCT, 'haar')
    coeffs_B_O_DCT_DWT = pywt.dwt2(B_O_DCT, 'haar')

    coeffs_R_W_DCT_DWT = pywt.dwt2(R_W_DCT, 'haar')
    coeffs_G_W_DCT_DWT = pywt.dwt2(G_W_DCT, 'haar')
    coeffs_B_W_DCT_DWT = pywt.dwt2(B_W_DCT, 'haar')

    #extract W_R_ext from DWT transformed R color component of the watermark
    cA_R_O, (cH_R_O, cV_R_O, cD_R_O) = coeffs_R_O_DCT_DWT
    cA_R_W, (cH_R_W, cV_R_W, cD_R_W) = coeffs_R_W_DCT_DWT

    #extract reverse of the embedding of  W_RA, W_RB, W_RC, W_RD into LL, LH, HL, HH 
    extracted_W_RA_resized = (cA_R_W - cA_R_O) / scaling_factor
    extracted_W_RB_resized = (cH_R_W - cH_R_O) / scaling_factor
    extracted_W_RC_resized = (cV_R_W - cV_R_O) / scaling_factor
    extracted_W_RD_resized = (cD_R_W - cD_R_O) / scaling_factor

    #merge the extracted watermark sub-blocks to obtain W_R_ext,DCT
    recon_h = cA_R_O.shape[0] * 2
    recon_w = cA_R_O.shape[1] * 2

    W_R_ext_DCT_reconstructed = np.zeros((recon_h, recon_w), dtype=np.float32)
    mid_h, mid_w = recon_h // 2, recon_w // 2
    # Ensure sizes match before placing
    W_R_ext_DCT_reconstructed[0:mid_h, 0:mid_w] = cv2.resize(extracted_W_RA_resized, (mid_w, mid_h), interpolation=cv2.INTER_LINEAR)
    W_R_ext_DCT_reconstructed[0:mid_h, mid_w:] = cv2.resize(extracted_W_RB_resized, (recon_w - mid_w, mid_h), interpolation=cv2.INTER_LINEAR)
    W_R_ext_DCT_reconstructed[mid_h:, 0:mid_w] = cv2.resize(extracted_W_RC_resized, (mid_w, recon_h - mid_h), interpolation=cv2.INTER_LINEAR)
    W_R_ext_DCT_reconstructed[mid_h:, mid_w:] = cv2.resize(extracted_W_RD_resized, (recon_w - mid_w, recon_h - mid_h), interpolation=cv2.INTER_LINEAR)

    W_R_ext_DCT = W_R_ext_DCT_reconstructed # This is the extracted DCT of the R watermark
    #IDCR to obtain the W_R_ext
    W_R_ext = block_idct2d(W_R_ext_DCT, block_size)

    W_R_ext_norm = cv2.normalize(W_R_ext, None, 0, 255, cv2.NORM_MINMAX)
    W_R_ext_uint8 = W_R_ext_norm.astype(np.uint8)
    
    _, W_R_ext_binary = cv2.threshold(W_R_ext_uint8, 99, 255, cv2.THRESH_BINARY)


    #extract W_G_ext from DWT transformed G color component of the watermark
    cA_G_O, (cH_G_O, cV_G_O, cD_G_O) = coeffs_G_O_DCT_DWT
    cA_G_W, (cH_G_W, cV_G_W, cD_G_W) = coeffs_G_W_DCT_DWT

    #extract reverse of the embedding of  W_RA, W_RB, W_RC, W_RD into LL, LH, HL, HH 
    extracted_W_GA_resized = (cA_G_W - cA_G_O) / scaling_factor
    extracted_W_GB_resized = (cH_G_W - cH_G_O) / scaling_factor
    extracted_W_GC_resized = (cV_G_W - cV_G_O) / scaling_factor
    extracted_W_GD_resized = (cD_G_W - cD_G_O) / scaling_factor

    #merge the extracted watermark sub-blocks to obtain W_R_ext,DCT
    recon_h = cA_G_O.shape[0] * 2
    recon_w = cA_G_O.shape[1] * 2

    W_G_ext_DCT_reconstructed = np.zeros((recon_h, recon_w), dtype=np.float32)
    mid_h, mid_w = recon_h // 2, recon_w // 2
    # Ensure sizes match before placing
    W_G_ext_DCT_reconstructed[0:mid_h, 0:mid_w] = cv2.resize(extracted_W_GA_resized, (mid_w, mid_h), interpolation=cv2.INTER_LINEAR)
    W_G_ext_DCT_reconstructed[0:mid_h, mid_w:] = cv2.resize(extracted_W_GB_resized, (recon_w - mid_w, mid_h), interpolation=cv2.INTER_LINEAR)
    W_G_ext_DCT_reconstructed[mid_h:, 0:mid_w] = cv2.resize(extracted_W_GC_resized, (mid_w, recon_h - mid_h), interpolation=cv2.INTER_LINEAR)
    W_G_ext_DCT_reconstructed[mid_h:, mid_w:] = cv2.resize(extracted_W_GD_resized, (recon_w - mid_w, recon_h - mid_h), interpolation=cv2.INTER_LINEAR)

    W_G_ext_DCT = W_G_ext_DCT_reconstructed # This is the extracted DCT of the R watermark
    #IDCR to obtain the W_R_ext
    W_G_ext = block_idct2d(W_G_ext_DCT, block_size)
    
    W_G_ext_norm = cv2.normalize(W_G_ext, None, 0, 255, cv2.NORM_MINMAX)
    W_G_ext_uint8 = W_G_ext_norm.astype(np.uint8)
    _, W_G_ext_binary = cv2.threshold(W_G_ext_uint8, 99, 255, cv2.THRESH_BINARY)

    #extract W_B_ext from DWT transformed B color component of the watermark
    cA_B_O, (cH_B_O, cV_B_O, cD_B_O) = coeffs_B_O_DCT_DWT
    cA_B_W, (cH_B_W, cV_B_W, cD_B_W) = coeffs_B_W_DCT_DWT

    #extract reverse of the embedding of  W_RA, W_RB, W_RC, W_RD into LL, LH, HL, HH 
    extracted_W_BA_resized = (cA_B_W - cA_B_O) / scaling_factor
    extracted_W_BB_resized = (cH_B_W - cH_B_O) / scaling_factor
    extracted_W_BC_resized = (cV_B_W - cV_B_O) / scaling_factor
    extracted_W_BD_resized = (cD_B_W - cD_B_O) / scaling_factor

    #merge the extracted watermark sub-blocks to obtain W_R_ext,DCT
    recon_h = cA_B_O.shape[0] * 2
    recon_w = cA_B_O.shape[1] * 2

    W_B_ext_DCT_reconstructed = np.zeros((recon_h, recon_w), dtype=np.float32)
    mid_h, mid_w = recon_h // 2, recon_w // 2
    # Ensure sizes match before placing
    W_B_ext_DCT_reconstructed[0:mid_h, 0:mid_w] = cv2.resize(extracted_W_BA_resized, (mid_w, mid_h), interpolation=cv2.INTER_LINEAR)
    W_B_ext_DCT_reconstructed[0:mid_h, mid_w:] = cv2.resize(extracted_W_BB_resized, (recon_w - mid_w, mid_h), interpolation=cv2.INTER_LINEAR)
    W_B_ext_DCT_reconstructed[mid_h:, 0:mid_w] = cv2.resize(extracted_W_BC_resized, (mid_w, recon_h - mid_h), interpolation=cv2.INTER_LINEAR)
    W_B_ext_DCT_reconstructed[mid_h:, mid_w:] = cv2.resize(extracted_W_BD_resized, (recon_w - mid_w, recon_h - mid_h), interpolation=cv2.INTER_LINEAR)

    W_B_ext_DCT = W_B_ext_DCT_reconstructed # This is the extracted DCT of the R watermark
    #IDCR to obtain the W_R_ext
    W_B_ext = block_idct2d(W_B_ext_DCT, block_size)
   
    W_B_ext_norm = cv2.normalize(W_B_ext, None, 0, 255, cv2.NORM_MINMAX)
    W_B_ext_uint8 = W_B_ext_norm.astype(np.uint8)
    _, W_B_ext_binary = cv2.threshold(W_B_ext_uint8, 99, 255, cv2.THRESH_BINARY)
    

    # W_bin_ext_float is the result from the inverse DCT

    W_bin_ext_float = cv2.merge([W_B_ext_binary, W_G_ext_binary, W_R_ext_binary])


    #W_ext = inverse_arnold_transform(W_S_ext, original_watermark_shape, arnold_iterations)
    W_ext = inverse_arnold_transform(W_bin_ext_float, original_watermark_shape, arnold_iterations)
    #W_ext = convert_binary_to_grayscale(W_ext)
    #W_ext = np.clip(W_ext, 0, 255).astype(np.uint8)

    return W_ext



if __name__ == "__main__":
    original_cover_path = "/home/chinasa/python_projects/watermark/images/sample.png"
    watermarked_image_path = "/home/chinasa/python_projects/watermark/output/watermarked_image.png"
    # This must match the original watermark's dimensions
    original_watermark_shape = (512, 512)
    original_watermark_path = "/home/chinasa/python_projects/watermark/images/watermark.png" # For comparison

    scaling_factor = 0.5 # Must be same as embedding
    arnold_iterations = 1 # Must be same as embedding

    try:
        extracted_watermark = watermark_extraction_process(
            original_cover_path,
            watermarked_image_path,
            scaling_factor,
            arnold_iterations,
            original_watermark_shape
        )

        output_path = "/home/chinasa/python_projects/watermark/output/extracted_watermark.png"
        cv2.imwrite(output_path, extracted_watermark)
        print(f"Watermark extraction complete. Extracted watermark saved to {output_path}")

        # Optional: Display the images
        original_watermark = cv2.imread(original_watermark_path, cv2.IMREAD_GRAYSCALE)
        if original_watermark is not None:
            cv2.imshow("Original Watermark", original_watermark)
        cv2.imshow("Extracted Watermark", extracted_watermark)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # You might want to calculate PSNR or NCC between original and extracted watermark
        # to evaluate performance.
        if original_watermark is not None:
            # Resize extracted to match original for comparison if necessary
            extracted_resized = cv2.resize(extracted_watermark, (original_watermark.shape[1], original_watermark.shape[0]), interpolation=cv2.INTER_AREA)
            # Calculate Mean Squared Error (MSE)
            mse = np.mean((original_watermark - extracted_resized) ** 2)
            if mse == 0:
                print("Perfect extraction!")
            else:
                max_pixel = 255.0
                psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
                print(f"PSNR between original and extracted watermark: {psnr:.2f} dB")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have OpenCV, NumPy, and PyWavelets installed (`pip install opencv-python numpy pywavelets`).")
        print("Also, verify your image paths and the precise implementation of sub-functions match the embedding process.")

