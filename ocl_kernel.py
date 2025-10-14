# OpenCL kernel for Sobel edge detection
# get_global_id(0) and get_global_id(1) retrieve the current pixel's indices, allowing parallel processing of each pixel.
def ocl_kernel():
    return """
__kernel void sobel_filter(__global const uchar *input_image,
                            __global uchar *output_image,
                            const int width,
                            const int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int gx = -input_image[(y-1) * width + (x-1)] + input_image[(y-1) * width + (x+1)]
                 -2 * input_image[y * width + (x-1)] + 2 * input_image[y * width + (x+1)]
                 -input_image[(y+1) * width + (x-1)] + input_image[(y+1) * width + (x+1)];
        
        int gy = input_image[(y-1) * width + (x-1)] + 2 * input_image[(y-1) * width + x] + input_image[(y-1) * width + (x+1)]
                 -input_image[(y+1) * width + (x-1)] - 2 * input_image[(y+1) * width + x] - input_image[(y+1) * width + (x+1)];

        int g = (int)sqrt((float)(gx * gx + gy * gy));
        output_image[y * width + x] = (g > 255) ? 255 : g;
    }
}
"""