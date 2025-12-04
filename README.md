# SentinelLandsatPyTorch


Leveraging Deep Learning Frameworks for Geospatial Analysis: Identifying Natural Sources of Clean Water Using TensorFlow and PyTorch
Whitepaper


Author: Vamsha Shetty w CoPi | Senior Automation Architect/Sw Test Engineering Mgr

Date: December 3, 2025

Location: Bengaluru

<img width="754" height="314" alt="image" src="https://github.com/user-attachments/assets/5199215a-6011-418e-9688-ad89b4e55cd4" />


Abstract -

This whitepaper explores practical applications of deep learning frameworks—TensorFlow and PyTorch—in space science and geoscience for satellite image classification and land cover mapping. The primary objective is to identify natural sources of clean water (e.g., rivers, lakes, wetlands, snow/ice-fed springs) directly from remote sensing data, without focusing on post-filtration or treatment parameters. We present data sources, preprocessing pipelines, model architectures, training and evaluation strategies, case studies, deployment patterns, and future directions tailored to scalable geospatial analysis.


1. Introduction
Access to clean water is a foundational requirement for health, agriculture, and sustainable development. Rapid urbanization, climate variability, and land-use changes complicate the task of locating pristine water sources. Remote sensing offers synoptic, repeatable observations over large areas, enabling the detection and monitoring of surface water bodies and related hydrological features. Deep learning has emerged as a powerful approach for discriminating water from other land cover types, learning complex spectral–spatial patterns across multispectral and radar datasets, and scaling inference to national or global extents.
This paper focuses on the use of convolutional neural networks (CNNs) and modern segmentation architectures implemented in TensorFlow and PyTorch to identify likely clean water sources. We emphasize optical multispectral sensors (e.g., Sentinel-2, Landsat) and discuss augmentation with radar (SAR) for all-weather robustness. Our scope excludes water quality treatment or post-processing chemistry; we instead concentrate on detecting naturally occurring water sources and differentiating them from anthropogenic artifacts.


2. Data Sources
Key satellite platforms and products used for water detection:
- Sentinel-2 MSI: 10–60 m resolution with visible (RGB), NIR, and SWIR bands; ideal for water vs. land discrimination and turbidity proxies.
- Landsat 8/9 OLI: 30 m resolution with similar spectral coverage; long historical record for temporal analyses.
- MODIS: Coarser resolution but high temporal frequency for seasonal water dynamics.
- SAR (Sentinel-1): C-band radar, insensitive to cloud cover; complements optical data for all-weather detection of smooth water surfaces.
2.1 Spectral Indices Useful for Water Detection
- NDWI (Normalized Difference Water Index): Enhances open water using green and NIR bands.
- MNDWI (Modified NDWI): Uses SWIR to reduce vegetation/soil confusion in built-up areas.
- AWEI (Automated Water Extraction Index): Combines multiple bands to improve separation of shadows and dark surfaces from water.
2.2 Preprocessing Pipeline
- Atmospheric correction (e.g., surface reflectance products).
- Cloud and shadow masking (e.g., QA bands, cloud probability masks).
- Radiometric normalization and per-band scaling.
- Spatial alignment and resampling among sensors.
- Tiling and patch generation for model training.

  
3. Methodology
We explore two complementary approaches: TensorFlow-centric and PyTorch-centric pipelines for water detection and land cover mapping.
3.1 TensorFlow Applications
- Semantic segmentation with U-Net to classify water vs. non-water at pixel level.
- Transfer learning with EfficientNet or MobileNet for patch-level classification when segmentation labels are limited.
- Integration with platforms like Google Earth Engine (via exported tiles) for scalable data preparation and ingestion.
3.1.1 TensorFlow U-Net (Multispectral Input)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Example U-Net for 6-band input (e.g., RGB, NIR, SWIR1, SWIR2)
INPUT_BANDS = 6
NUM_CLASSES = 2  # water, non-water

inputs = keras.Input(shape=(256, 256, INPUT_BANDS))

# Encoder
c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
p1 = layers.MaxPooling2D((2, 2))(c1)

c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
p2 = layers.MaxPooling2D((2, 2))(c2)

c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)
p3 = layers.MaxPooling2D((2, 2))(c3)

c4 = layers.Conv2D(256, 3, activation='relu', padding='same')(p3)
c4 = layers.Conv2D(256, 3, activation='relu', padding='same')(c4)
p4 = layers.MaxPooling2D((2, 2))(c4)

# Bottleneck
bn = layers.Conv2D(512, 3, activation='relu', padding='same')(p4)
bn = layers.Conv2D(512, 3, activation='relu', padding='same')(bn)

# Decoder
u1 = layers.UpSampling2D((2, 2))(bn)
u1 = layers.Concatenate()([u1, c4])
d1 = layers.Conv2D(256, 3, activation='relu', padding='same')(u1)
d1 = layers.Conv2D(256, 3, activation='relu', padding='same')(d1)

u2 = layers.UpSampling2D((2, 2))(d1)
u2 = layers.Concatenate()([u2, c3])
d2 = layers.Conv2D(128, 3, activation='relu', padding='same')(u2)
d2 = layers.Conv2D(128, 3, activation='relu', padding='same')(d2)

u3 = layers.UpSampling2D((2, 2))(d2)
u3 = layers.Concatenate()([u3, c2])
d3 = layers.Conv2D(64, 3, activation='relu', padding='same')(u3)
d3 = layers.Conv2D(64, 3, activation='relu', padding='same')(d3)

u4 = layers.UpSampling2D((2, 2))(d3)
u4 = layers.Concatenate()([u4, c1])
d4 = layers.Conv2D(32, 3, activation='relu', padding='same')(u4)
d4 = layers.Conv2D(32, 3, activation='relu', padding='same')(d4)

outputs = layers.Conv2D(NUM_CLASSES, 1, activation='softmax')(d4)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Data pipeline would map tiles -> (H, W, 6) arrays and masks -> (H, W, 1) labels
# Example training: model.fit(train_ds, validation_data=val_ds, epochs=50)

3.2 PyTorch Applications
- Pixel-wise segmentation with DeepLabV3+ or U-Net variants using torchvision or segmentation_models.pytorch.
- Custom CNNs for spectral fusion (e.g., 6–10 band inputs) and anomaly detection of ephemeral water.
- Use TorchGeo to handle geospatial datasets, tiling, and spatial indexing.
3.2.1 PyTorch DeepLabV3+ (Multispectral Input)

import torch
import torch.nn as nn
import torchvision

# Example: adapt DeepLabV3 with a custom input channel count (e.g., 6 bands)
IN_CHANNELS = 6
NUM_CLASSES = 2

class MultiSpectralResNet(torch.nn.Module):
    def __init__(self, in_channels=IN_CHANNELS):
        super().__init__()
        # Start from ResNet-50 and replace first conv
        self.backbone = torchvision.models.resnet50(weights=None)
        self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        # Forward up to the C5 feature map used by DeepLab
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

class SimpleDeepLabHead(nn.Module):
    def __init__(self, in_channels=2048, num_classes=NUM_CLASSES):
        super().__init__()
        self.aspp = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        x = self.aspp(x)
        x = self.classifier(x)
        return x

class DeepLabLike(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = MultiSpectralResNet(in_channels)
        self.head = SimpleDeepLabHead(2048, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        # Upsample to input size
        logits = nn.functional.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return logits

# Example usage
model = DeepLabLike(IN_CHANNELS, NUM_CLASSES)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# for batch in dataloader:
#     imgs, masks = batch  # imgs: (B, 6, H, W); masks: (B, H, W)
#     optimizer.zero_grad()
#     logits = model(imgs)
#     loss = criterion(logits, masks)
#     loss.backward()
#     optimizer.step()


4. Training and Evaluation
- Split data into train/validation/test by geographic tiles to avoid spatial leakage.
- Use class-balanced sampling; water pixels are often minority in large scenes.
- Metrics: Intersection-over-Union (IoU) for water class, F1-score, Precision/Recall, and boundary accuracy to quantify shoreline delineation.
- Employ multi-temporal training (seasonal stacks) to improve robustness to phenology and hydrological variability.
- Data augmentation: random rotations, flips, brightness/contrast jitter per band, synthetic cloud overlays, and cutout to simulate occlusions.

  
5. Case Studies
5.1 Himalayan Glacier-Fed Water Detection (PyTorch): Train DeepLab-like models on summer vs. winter optical scenes plus Sentinel-1 SAR to distinguish snow/ice, proglacial lakes, and meltwater channels. Multi-modal fusion reduces confusion between snow shadows and dark water.
5.2 Amazon Basin River Network Mapping (TensorFlow): U-Net segmentation on Sentinel-2 mosaics with MNDWI thresholding as weak labels. Post-process predictions with morphological operations to produce connected river graphs for navigation and planning.
5.3 Urban Reservoir and Wetland Identification (Multispectral Fusion): Combine SWIR with NIR to reduce rooftop/asphalt false positives. Employ object-based filtering to remove small dark features that are unlikely to be water (e.g., parking lots).

   
7. Deployment and Scalability
- Batch inference on cloud GPUs (Azure, AWS, GCP) over tiled datasets stored in cloud object storage (e.g., Cloud-Optimized GeoTIFFs).
- Use sliding-window inference with overlap to minimize edge artifacts; stitch predictions using weighted blending.
- Integrate outputs into GIS systems (GeoPackage, PostGIS) for visualization, querying, and downstream decision-making.
- Build alerting pipelines that flag new or expanding water bodies from time-series analysis (e.g., flooding, seasonal wetlands).
7. Future Directions
- Incorporate SAR (Sentinel-1) more systematically for cloud-prone regions; explore polarimetric features and coherence for water surface smoothness.
- Multi-modal learning: fuse optical + SAR + DEM (topography) to prioritize likely clean sources (elevational springs, glacial lakes) and filter out urban runoff.
- Uncertainty quantification (MC Dropout or ensembles) to produce confidence maps that aid field validation and resource allocation.
- Active learning with human-in-the-loop labeling to refine models in novel geographies.

  
8. Limitations and Ethical Considerations
- Water detected via remote sensing reflects surface presence and spectral signatures; it does not guarantee potability. Field validation remains essential.
- Seasonal and atmospheric variability (haze, sun glint) can affect detection; radar helps but introduces its own complexities.
- Publishing precise locations of pristine water sources may have ecological and social implications; consider access control and responsible disclosure.

<img width="1536" height="1024" alt="Designer (1)" src="https://github.com/user-attachments/assets/aff17e95-a41c-4b63-b801-61230438842f" />


9. Conclusion
TensorFlow and PyTorch provide robust, flexible tooling for large-scale geospatial analysis. Using segmentation architectures (U-Net, DeepLab variants), multispectral inputs, and thoughtful preprocessing, practitioners can accurately delineate natural water sources across diverse terrains. The outlined pipelines and examples demonstrate how to operationalize these methods for monitoring, planning, and sustainability initiatives.

Credits: CoPilot Analysis


11. References
- Sentinel-2 MSI User Guide; Landsat 8/9 OLI Instrument Guides
- McFeeters, S.K. (1996). The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features.
- Xu, H. (2006). Modification of normalised difference water index (NDWI) to enhance open water features: A comparison of Landsat imagery.
- TorchGeo documentation; TensorFlow/Keras documentation; torchvision models and segmentation references
