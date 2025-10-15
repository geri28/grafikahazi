import cv2
import os
import time
import numpy as np 
from numba import jit
import pyopencl as cl
from ocl_kernel import *
# import cupy as cp --> fúj
from concurrent.futures import ThreadPoolExecutor

ocl_kernelString = ocl_kernel()

sobel_x = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])

sobel_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])



def edge_detection(image_path):
    start_image_process = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_blur = cv2.GaussianBlur(image, (3,3), sigmaX=0, sigmaY=0)
    end_image_process = time.time() - start_image_process


    start_actual_process = time.time()
    sobel_x = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3) 
    sobel_y = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)  
    edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    end_actual_process = time.time() - start_actual_process
    return edges, end_image_process, end_actual_process

@jit(nopython=True)
def simd_sobel_edge_detection(image):
    height, width = image.shape
    output = np.zeros((height, width), dtype=np.float64)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            gx = 0.0
            gy = 0.0

            for row in range(-1, 2):
                for column in range(-1, 2):
                    gx += sobel_x[row + 1, column + 1] * image[i + row, j + column]
                    gy += sobel_y[row + 1, column + 1] * image[i + row, j + column]
            output[i, j] = (gx ** 2 + gy ** 2) ** 0.5

    return output


def simd_edge_detection(image_path):
    start_image_process = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image, (3, 3), sigmaX=0, sigmaY=0)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")

    end_image_process = time.time() - start_image_process
    start_actual_process = time.time()
    edges = simd_sobel_edge_detection(image)
    end_actual_process = time.time() - start_actual_process
    return edges, end_image_process, end_actual_process


def gpgpu_edge_detection(image_path):

    start_image_process = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    height, width = image.shape
    image = cv2.GaussianBlur(image, (3, 3), sigmaX=0, sigmaY=0)

    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")

    output_image = np.zeros_like(image)
    end_image_process = time.time() - start_image_process

    start_actual_process = time.time()

    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    mf = cl.mem_flags
    input_buffer = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image)
    output_buffer = cl.Buffer(context, mf.WRITE_ONLY, output_image.nbytes)

    program = cl.Program(context, ocl_kernelString).build()

    global_size = (width, height)
    program.sobel_filter(queue, global_size, None, input_buffer, output_buffer, np.int32(width), np.int32(height))

    cl.enqueue_copy(queue, output_image, output_buffer)

    end_actual_process = time.time() - start_actual_process

    return output_image, end_image_process, end_actual_process

def single_thread_processing(image_paths, method='opencv'):
    results = []
    timeVals = dict()
    timeVals["Img"] = []
    timeVals["Act"] = []
    start_time = time.time()  
    for path in image_paths:
        if method == 'opencv':
            edges, end_image_process, end_actual_process = edge_detection(path)
        elif method == 'simd':
            edges, end_image_process, end_actual_process = simd_edge_detection(path)
        elif method == 'gpgpu':
            edges, end_image_process, end_actual_process = gpgpu_edge_detection(path)
        results.append((path, edges))
        timeVals["Img"].append(end_image_process)
        timeVals["Act"].append(end_actual_process)
    elapsed_time = time.time() - start_time
    return results, elapsed_time, timeVals


def multi_thread_processing(image_paths, method='opencv'):
    results = []
    timeVals = dict()
    timeVals["Img"] = []
    timeVals["Act"] = []
    start_time = time.time()  
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(
            edge_detection if method == 'opencv' else simd_edge_detection if method == 'simd' else gpgpu_edge_detection,
            path): path for path in image_paths}
        for future in futures:
            edges, end_image_process, end_actual_process = future.result()
            results.append((futures[future], edges))
        timeVals["Img"].append(end_image_process)
        timeVals["Act"].append(end_actual_process)
    elapsed_time = time.time() - start_time
    return results, elapsed_time, timeVals


def save_results(results, method):
    output_dir = f'images/{method}'
    os.makedirs(output_dir, exist_ok=True)
    for path, edges in results:
        filename = os.path.basename(path)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, edges)


def load_images_from_directory(directory):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(image_extensions)]

def describeNumbersList(numbers):
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)

    if n % 2 == 0:
        median = (sorted_numbers[n // 2 - 1] + sorted_numbers[n // 2]) / 2
    else:
        median = sorted_numbers[n // 2]

    average = sum(sorted_numbers) / n

    min_value = sorted_numbers[0]
    max_value = sorted_numbers[-1]

    return {
        "median": sec_to_ms(round(median, 5)),
        "average": sec_to_ms(round(average,5)),
        "min": sec_to_ms(round(min_value,5)),
        "max": sec_to_ms(round(max_value,5))
    }

def sec_to_ms(seconds):
    return f"{"" if seconds < 1 else str(seconds // 1) + "sec"} {str(int(seconds * 1_000)) + " ms" }"

def main():

    input_directory = "E:\munka\egyetem\MSc\hoffmann\images"

    image_paths = load_images_from_directory(input_directory)

    if not image_paths:
        print("No images found in the specified directory.")
        return

    methods = ['opencv', 'simd', 'gpgpu']
    results = {}

    for method in methods:
        single_thread_results, single_thread_time, single_timeDict = single_thread_processing(image_paths, method)
        multi_thread_results, multi_thread_time, multi_timeDict = multi_thread_processing(image_paths, method)


        results[method] = {
            'single_thread': (single_thread_results, single_thread_time, single_timeDict),
            'multi_thread': (multi_thread_results, multi_thread_time, multi_timeDict )
        }

        save_results(single_thread_results, f'SingleThread_{method}')
        save_results(multi_thread_results, f'MultiThread_{method}')

        print(f"Method: {method.capitalize()}\t-\tSingle Thread processing time:\t {sec_to_ms(single_thread_time)}")
        print(f"Method: {method.capitalize()}\t-\tMulti Thread processing time:\t {sec_to_ms(multi_thread_time)}")

    print("\nEdge detection completed and results saved.\n\nDetailed Overview\n")

    for meth, wholeLine in results.items():
        for threadName, valList in wholeLine.items():
            print(f"Method: {meth.capitalize()}\t|\tThread: {threadName}\t==>\tImg Sum: {sec_to_ms(np.sum(valList[-1]["Img"]))}, Act Sum: {sec_to_ms(np.sum(valList[-1]["Act"]))}")
            print(f"Act: {describeNumbersList(valList[-1]["Act"])}")
            print(f"Img: {describeNumbersList(valList[-1]["Img"])}\n")

if __name__ == "__main__":
    main()