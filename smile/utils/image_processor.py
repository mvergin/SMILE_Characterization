from typing import List, Optional
from PIL import Image
import numpy as np


class ImageProcessor:
    """
    ImageProcessor class is designed to process images for display on micro displays.
    The class allows loading images, converting them to a specified grayscale format,
    cropping them to a suitable size, and transforming them to a format suitable for
    micro displays.

    The processed data, named `display_data`, is a list of lists, where each inner
    list consists of eight 128-bit numbers that represent the pixel information
    from the source image. This format is optimized for sending to a 256x256 micro display.

    Attributes:
    -----------
    _img: PIL.Image.Image
        The loaded and processed image in the PIL format.

    Methods:
    --------
    load_image(path: str) -> None:
        Loads an image from the specified path into the ImageProcessor instance.

    process(
        canvas_width: int = 256,
        canvas_height: int = 256,
        grayscale_levels: int = 16,
        invert: bool = False,
        rotate: int = 0,
    ) -> None:
        Processes the loaded image, converting it to the desired grayscale format,
        optionally inverting the grayscale levels, rotating it, and centering it within a canvas of the desired dimensions.

    display_data -> List[List[int]]:
        Property that returns the processed image in a format suitable for the micro display.

    Example:
    --------
    processor = ImageProcessor("path_to_image.jpg")
    processor.process(invert=True, rotate=90)
    data_for_display = processor.display_data
    """

    def __init__(self, path: Optional[str] = None):
        """
        Initialize the ImageProcessor.

        Args:
            path (Optional[str]): Path to the image file. If provided, the image is loaded on initialization.
        """
        self._img: Optional[Image.Image] = None
        if path:
            self.load_image(path)

        self.mapping_8lvl = {0: 0, 1: 1, 2: 3, 3: 5, 4: 7, 5: 8, 6: 11, 7: 14}

    def load_image(self, path: str) -> None:
        """
        Loads an image from the given path.

        Args:
            path (str): Path to the image file to be loaded.
        Raises:
            ValueError: If the file is not found or is not a recognized image format.
        """
        try:
            self._img = Image.open(path)
        except FileNotFoundError:
            raise ValueError(f"The file at {path} was not found.")
        except Image.UnidentifiedImageError:
            raise ValueError("The provided file is not a recognized image format.")

    def _convert_to_nbit_grayscale(
        self, mapping: dict = None, levels: int = 16
    ) -> Image.Image:
        """
        Convert the image to n-bit grayscale.

        Args:
            levels (int): Number of grayscale levels. Default is 16 for 4-bit grayscale.
        Returns:
            Image.Image: An image in n-bit grayscale.
        """
        if self._img.mode != "L":
            self._img = self._img.convert("L")

        factor = (levels - 1) / 255
        if mapping is None:
            img_nbit = Image.eval(self._img, lambda x: int(x * factor))
        else:
            img_nbit = Image.eval(self._img, lambda x: mapping[int(x * factor)])

        return img_nbit

    def _center_within_canvas(
        self, canvas_width: int, canvas_height: int
    ) -> Image.Image:
        """
        Center the image within a canvas of the given dimensions. If the image is smaller than the canvas,
        it is placed at the center with the rest filled with zeros.

        Args:
            canvas_width (int): Width of the canvas.
            canvas_height (int): Height of the canvas.
        Returns:
            Image.Image: Centered image.
        """
        centered_img = Image.new("L", (canvas_width, canvas_height), 0)
        left = (canvas_width - self._img.width) // 2
        top = (canvas_height - self._img.height) // 2
        centered_img.paste(self._img, (left, top))
        return centered_img

    def _chunks_of_32(
        self, pixel_list: List[int], full_display: bool = False
    ) -> List[int]:
        """
        Given a list of pixel values, process them in chunks of 32 to produce 128-bit numbers,
        with optional intercalation for a full display.

        Args:
            pixel_list (List[int]): List of pixel values.
            full_display (bool): Whether the processing is for the full display or just a quarter.

        Returns:
            List[int]: List of 128-bit numbers created by chunking the pixel values, intercalated if needed.
        """
        chunked_values = []

        if not full_display:
            # for a quarter
            for row in range(0, len(pixel_list), 256):
                for group in range(7, -1, -1):  # col group
                    chunk = pixel_list[row + group * 32 : row + (group + 1) * 32]
                    chunk_value = sum(
                        val << (4 * (31 - idx)) for idx, val in enumerate(chunk)
                    )
                    chunked_values.append(chunk_value)
        else:
            quadrant_size = 256 * 256
            quadrants = [
                pixel_list[:quadrant_size],  # top left
                pixel_list[quadrant_size : quadrant_size * 2],  # top right
                pixel_list[quadrant_size * 2 : quadrant_size * 3],  # bottom left
                pixel_list[quadrant_size * 3 :],  # bottom right
            ]

            for row in range(256):  # row
                for group in range(7, -1, -1):  # Procesar grupos en orden invertido
                    for quadrant in quadrants:  # Intercalar cuadrantes
                        start_idx = row * 256 + group * 32
                        end_idx = start_idx + 32
                        chunk = quadrant[start_idx:end_idx]
                        chunk_value = sum(
                            val << (4 * (31 - idx)) for idx, val in enumerate(chunk)
                        )
                        chunked_values.append(chunk_value)

        return chunked_values

    # def _chunks_of_32(self, pixel_list: List[int]) -> List[int]:
    #     """
    #     Given a list of pixel values, process them in chunks of 32 to produce 128-bit numbers.
    #
    #     Args:
    #         pixel_list (List[int]): List of pixel values.
    #     Returns:
    #         List[int]: List of 128-bit numbers created by chunking the pixel values.
    #     """
    #     chunked_values = []
    #     for i in range(0, len(pixel_list), 256):
    #         for j in range(7, -1, -1):
    #             chunk = pixel_list[i + j*32: i + (j+1)*32]
    #             chunk_value = sum(val << (4 * (31 - idx)) for idx, val in enumerate(chunk))
    #             chunked_values.append(chunk_value)
    #     return chunked_values

    def process(
        self,
        canvas_width: int = 256,
        canvas_height: int = 256,
        grayscale_levels: int = 16,
        invert: bool = False,
        rotate: int = 0,
    ) -> None:
        """
        Process the image by converting to n-bit grayscale, optionally inverting the grayscale levels,
        centering it within the canvas, and then converting it to a matrix B as described.

        Args:
            canvas_width (int): Width of the final canvas.
            canvas_height (int): Height of the final canvas.
            grayscale_levels (int): Number of grayscale levels for conversion.
            invert (bool): If True, invert the grayscale values.
            rotate (int): Degrees to rotate the image prior to centering.
        Raises:
            ValueError: If no image is loaded.
        """
        if not self._img:
            raise ValueError("No image loaded. Use load_image(path) to load an image.")

        if grayscale_levels == 8:
            self._img = self._convert_to_nbit_grayscale(
                mapping=self.mapping_8lvl, levels=grayscale_levels
            )
        else:
            self._img = self._convert_to_nbit_grayscale(levels=grayscale_levels)

        if invert:
            self._img = Image.eval(self._img, lambda x: (grayscale_levels - 1) - x)

        if rotate:
            self._img = self._img.rotate(rotate, expand=True)

        self._img = self._center_within_canvas(canvas_width, canvas_height)

    @property
    def display_data(self) -> List[int]:
        """
        Get the processed image in a flattened format suitable for the micro display.

        Returns:
            List[int]: Flattened data for the micro display.
        Raises:
            ValueError: If no image has been processed yet.
        """
        if not self._img:
            raise ValueError(
                "No image has been processed yet. Use the process() method."
            )

        pixel_data = list(self._img.getdata())
        # Automatically detect if we are in 512x512 mode
        # 256x256 = 65,536 pixels
        # 512x512 = 262,144 pixels
        is_full_mode = len(pixel_data) == (512 * 512)

        return self._chunks_of_32(pixel_data, full_display=is_full_mode)

    def generate_pattern(self) -> List[int]:
        """
        Generate a test pattern of 16x16 blocks with varying grayscale values.

        Returns:
            List[int]: A flattened list of the test pattern.
        """
        width, height = 256, 256
        block_size = 64
        image_pattern = np.zeros((height, width))
        num_blocks = width // block_size

        values = np.arange(16).reshape(
            4, 4
        )  # 16 possible levels arranged in the desired order

        for i in range(num_blocks):
            for j in range(num_blocks):
                image_pattern[
                    i * block_size : (i + 1) * block_size,
                    j * block_size : (j + 1) * block_size,
                ] = values[i, j]

        return [int(val) for val in image_pattern.flatten().tolist()]

    @property
    def test_pattern(self) -> List[int]:
        """
        Property that retrieves a processed test pattern suitable for the micro display.

        Returns:
            List[int]: A list of processed test pattern values.
        """
        pattern = self.generate_pattern()
        return self._chunks_of_32(pattern)
