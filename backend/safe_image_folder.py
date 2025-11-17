from torchvision import datasets
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            path, target = self.samples[index]
            image = self.loader(path)

            # Convert palette images with transparency to RGBA
            if image.mode == "P":
                image = image.convert("RGBA")

            if self.transform:
                image = self.transform(image)
            return image, target

        except (OSError, ValueError) as e:
            print(f"Skipping corrupted image at index {index}: {e}")
            dummy_image = Image.new("RGB", (224, 224))
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, -1
