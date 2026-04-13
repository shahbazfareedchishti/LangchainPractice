from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

text = """The Samsung Galaxy A57 5G represents the latest iteration in the popular A series lineup, officially released on March 25, 2026. This device is positioned as a high-performance mid-range smartphone that bridges the gap between affordable accessibility and premium innovation. Its arrival marks a significant shift for the series, integrating advanced artificial intelligence features and a refined physical design that aligns with modern aesthetic standards.

In terms of physical design, the Galaxy A57 5G features a slim and lightweight profile, measuring only 6.9 millimeters in thickness and weighing 179 grams. It is constructed with a glass body and is rated IP68 for water and dust resistance, ensuring durability for everyday use. The front of the device is dominated by a 6.7 inch Super AMOLED Plus display which supports a 120Hz refresh rate and includes Vision Booster technology for better visibility under direct sunlight. This screen is protected by slim bezels that provide an immersive viewing experience for media consumption and gaming.

Internally, the smartphone is powered by the Exynos 1680 processor, which is supported by an upgraded CPU, GPU, and NPU designed to handle complex multitasking and AI-driven tasks. To maintain performance during intensive activities like gaming or high-resolution video recording, Samsung has included a vapor chamber that is 13 percent larger than previous models. The device comes in multiple configurations, offering up to 12GB of RAM and 512GB of internal storage. It runs on Android 16 with the One UI 8.5 interface, and Samsung has committed to providing six generations of OS upgrades and six years of security updates.

The camera system is another focal point, featuring a triple rear setup led by a 50 megapixel main sensor with an f/1.8 aperture. This is accompanied by a 12 megapixel ultra wide lens and a 5 megapixel macro camera. On the front, there is a 12 megapixel selfie camera. These hardware components work in tandem with the new Awesome Intelligence suite, which introduces tools such as Object Eraser 3 for removing distractions from photos and Best Face for group photography. Additionally, the software includes productivity tools like Voice Transcription for recording meetings and AI Select for easier text extraction from the screen.

Battery life is supported by a 5000mAh capacity, which remains consistent with the series' reputation for longevity. The device supports Super Fast Charging 2.0, allowing the battery to reach 60 percent in approximately 30 minutes. Security is managed through the Samsung Knox platform, and the phone includes an in-display fingerprint sensor for biometrics. Available in colors such as Awesome Navy, Awesome Icyblue, and Awesome Lilac, the Galaxy A57 5G seeks to offer a flagship-like experience at a more accessible price point, starting at approximately 549 dollars in the United States and 56,999 rupees in India.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=10
)

result = splitter.split_text(text)

print(result)