import cv2
import numpy as np
import pywt
from scipy.fftpack import dct, idct
from watermark_embedding import block_dct2d, block_idct2d

def inverse_arnold_transform(image, original_shape, iterations=1):
    if image.shape[0] != image.shape[1]:
        raise ValueError("Image must be square for the inverse Arnold transform.")
    n = image.shape[0]
    unscrambled_image = np.copy(image)
    for _ in range(iterations):
        original_image = np.zeros_like(unscrambled_image)
        for x_prime in range(n):
            for y_prime in range(n):
                x = (x_prime - y_prime) % n
                y = (-x_prime + 2 * y_prime) % n
                original_image[x, y] = unscrambled_image[x_prime, y_prime]
        unscrambled_image = original_image
    return cv2.resize(unscrambled_image, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_AREA)

def watermark_extraction_process(original_cover_path, watermarked_image_path, original_watermark_path, scaling_factor, arnold_iterations=1):
    # Load images
    I_O = cv2.imread(original_cover_path)
    if I_O is None:
        raise FileNotFoundError(f"Original cover image not found: {original_cover_path}")
    I_W = cv2.imread(watermarked_image_path)
    if I_W is None:
        raise FileNotFoundError(f"Watermarked image not found: {watermarked_image_path}")
    
    # --- GET ORIGINAL WATERMARK SHAPE FROM THE IMAGE FILE ---
    watermark_original = cv2.imread(original_watermark_path, cv2.IMREAD_GRAYSCALE)
    if watermark_original is None:
        raise FileNotFoundError(f"Original watermark image not found: {original_watermark_path}")
    original_watermark_shape = watermark_original.shape
    original_h, original_w = original_watermark_shape
    # --- END ---

    block_size = 8

    # Separate images into B, G, R components and perform DCT
    B_O, G_O, R_O = cv2.split(I_O)
    B_W, G_W, R_W = cv2.split(I_W)

    R_O_DCT = block_dct2d(R_O.astype(np.float32), block_size)
    G_O_DCT = block_dct2d(G_O.astype(np.float32), block_size)
    B_O_DCT = block_dct2d(B_O.astype(np.float32), block_size)
    R_W_DCT = block_dct2d(R_W.astype(np.float32), block_size)
    G_W_DCT = block_dct2d(G_W.astype(np.float32), block_size)
    B_W_DCT = block_dct2d(B_W.astype(np.float32), block_size)

    # Perform DWT on the DCT transformed color components
    coeffs_R_O_DCT_DWT = pywt.dwt2(R_O_DCT, 'haar')
    coeffs_G_O_DCT_DWT = pywt.dwt2(G_O_DCT, 'haar')
    coeffs_B_O_DCT_DWT = pywt.dwt2(B_O_DCT, 'haar')
    coeffs_R_W_DCT_DWT = pywt.dwt2(R_W_DCT, 'haar')
    coeffs_G_W_DCT_DWT = pywt.dwt2(G_W_DCT, 'haar')
    coeffs_B_W_DCT_DWT = pywt.dwt2(B_W_DCT, 'haar')

    def extract_channel_watermark(coeffs_O, coeffs_W, scaling_factor, original_watermark_shape):
        cA_O, (cH_O, cV_O, cD_O) = coeffs_O
        cA_W, (cH_W, cV_W, cD_W) = coeffs_W

        # Extract the signal by inverting the embedding formula
        extracted_W_A = (cA_W - cA_O) / scaling_factor
        extracted_W_B = (cH_W - cH_O) / scaling_factor
        extracted_W_C = (cV_W - cV_O) / scaling_factor
        extracted_W_D = (cD_W - cD_O) / scaling_factor

        # Reconstruct the full watermark DCT component using the correct dimensions
        original_h, original_w = original_watermark_shape
        recon_h, recon_w = original_h, original_w
        recon_mid_h, recon_mid_w = original_h // 2, original_w // 2

        W_ext_DCT_reconstructed = np.zeros((recon_h, recon_w), dtype=np.float32)
        W_ext_DCT_reconstructed[0:recon_mid_h, 0:recon_mid_w] = cv2.resize(extracted_W_A, (recon_mid_w, recon_mid_h), interpolation=cv2.INTER_LINEAR)
        W_ext_DCT_reconstructed[0:recon_mid_h, recon_mid_w:] = cv2.resize(extracted_W_B, (recon_w - recon_mid_w, recon_mid_h), interpolation=cv2.INTER_LINEAR)
        W_ext_DCT_reconstructed[recon_mid_h:, 0:recon_mid_w] = cv2.resize(extracted_W_C, (recon_mid_w, recon_h - recon_mid_h), interpolation=cv2.INTER_LINEAR)
        W_ext_DCT_reconstructed[recon_mid_h:, recon_mid_w:] = cv2.resize(extracted_W_D, (recon_w - recon_mid_w, recon_h - recon_mid_h), interpolation=cv2.INTER_LINEAR)   #INTER_LANCZOS4
        
        # Invert the DCT
        W_ext = block_idct2d(W_ext_DCT_reconstructed, block_size)
        return W_ext

    # Extract watermark from each channel using the helper function
    W_R_ext = extract_channel_watermark(coeffs_R_O_DCT_DWT, coeffs_R_W_DCT_DWT, scaling_factor, original_watermark_shape)
    W_G_ext = extract_channel_watermark(coeffs_G_O_DCT_DWT, coeffs_G_W_DCT_DWT, scaling_factor, original_watermark_shape)
    W_B_ext = extract_channel_watermark(coeffs_B_O_DCT_DWT, coeffs_B_W_DCT_DWT, scaling_factor, original_watermark_shape)

    # Average the three extracted channels to get a single, combined result
    W_ext_combined = (W_R_ext + W_G_ext + W_B_ext) / 3.0

    # Normalize and binarize the single-channel combined result
    W_norm = cv2.normalize(W_ext_combined, None, 0, 255, cv2.NORM_MINMAX)
    W_norm_uint8 = W_norm.astype(np.uint8)
    
    # Use the threshold you found to be effective, e.g., 99
    _, W_binary = cv2.threshold(W_norm_uint8, 128, 255, cv2.THRESH_BINARY)
    
    # Apply inverse Arnold transform to the clean binary watermark
    extracted_watermark = inverse_arnold_transform(W_binary, original_watermark_shape, arnold_iterations)

    return extracted_watermark

if __name__ == "__main__":
    original_cover_path = "/home/chinasa/python_projects/watermark/images/sample.png"
    watermarked_image_path = "/home/chinasa/python_projects/watermark/output/watermarked_image.png"
    original_watermark_path = "/home/chinasa/python_projects/watermark/images/watermark.png"
    
    # This must be the SAME scaling factor used for embedding!
    scaling_factor = 0.01 
    arnold_iterations = 3 

    try:
        extracted_watermark = watermark_extraction_process(
            original_cover_path,
            watermarked_image_path,
            original_watermark_path,
            scaling_factor,
            arnold_iterations
        )
        output_path = "/home/chinasa/python_projects/watermark/output/extracted_watermark.png"
        cv2.imwrite(output_path, extracted_watermark)
        print(f"Watermark extraction complete. Extracted watermark saved to {output_path}")

        original_watermark = cv2.imread(original_watermark_path, cv2.IMREAD_GRAYSCALE)
        if original_watermark is not None:
            cv2.imshow("Original Watermark", original_watermark)
        cv2.imshow("Extracted Watermark", extracted_watermark)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if original_watermark is not None:
            extracted_resized = cv2.resize(extracted_watermark, (original_watermark.shape[1], original_watermark.shape[0]), interpolation=cv2.INTER_AREA)
            mse = np.mean((original_watermark - extracted_resized) ** 2)
            if mse == 0:
                print("Perfect extraction!")
            else:
                max_pixel = 255.0
                psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
                print(f"PSNR between original and extracted watermark: {psnr:.2f} dB")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your image paths and all other parameters are correct.")