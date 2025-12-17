# SentinelLandsatPyTorch


Leveraging Deep Learning Frameworks for Geospatial Analysis: Identifying Natural Sources of Clean Water Using TensorFlow and PyTorch
Whitepaper


Author: Vamsha Shetty w CoPi | Senior Automation Architect/Sw Test Engineering Mgr
Date: December 3, 2025
Location: Bengaluru

<img width="754" height="314" alt="image" src="https://github.com/user-attachments/assets/5199215a-6011-418e-9688-ad89b4e55cd4" />


Abstract -

This whitepaper explores practical applications of deep learning frameworks—TensorFlow and PyTorch—in space science and geoscience for satellite image classification and land cover mapping. The primary objective is to identify natural sources of clean water (e.g., rivers, lakes, wetlands, snow/ice-fed springs) directly from remote sensing data, without focusing on post-filtration or treatment parameters. We present data sources, preprocessing pipelines, model architectures, training and evaluation strategies, case studies, deployment patterns, and future directions tailored to scalable geospatial analysis.


1. Introduction -
Access to clean water is a foundational requirement for health, agriculture, and sustainable development. Rapid urbanization, climate variability, and land-use changes complicate the task of locating pristine water sources. Remote sensing offers synoptic, repeatable observations over large areas, enabling the detection and monitoring of surface water bodies and related hydrological features. Deep learning has emerged as a powerful approach for discriminating water from other land cover types, learning complex spectral–spatial patterns across multispectral and radar datasets, and scaling inference to national or global extents.
This paper focuses on the use of convolutional neural networks (CNNs) and modern segmentation architectures implemented in TensorFlow and PyTorch to identify likely clean water sources. We emphasize optical multispectral sensors (e.g., Sentinel-2, Landsat) and discuss augmentation with radar (SAR) for all-weather robustness. Our scope excludes water quality treatment or post-processing chemistry; we instead concentrate on detecting naturally occurring water sources and differentiating them from anthropogenic artifacts.


2. Data Sources -
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

  
3. Methodology -
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


4. Training and Evaluation -
- Split data into train/validation/test by geographic tiles to avoid spatial leakage.
- Use class-balanced sampling; water pixels are often minority in large scenes.
- Metrics: Intersection-over-Union (IoU) for water class, F1-score, Precision/Recall, and boundary accuracy to quantify shoreline delineation.
- Employ multi-temporal training (seasonal stacks) to improve robustness to phenology and hydrological variability.
- Data augmentation: random rotations, flips, brightness/contrast jitter per band, synthetic cloud overlays, and cutout to simulate occlusions.

  
5. Case Studies -
5.1 Himalayan Glacier-Fed Water Detection (PyTorch): Train DeepLab-like models on summer vs. winter optical scenes plus Sentinel-1 SAR to distinguish snow/ice, proglacial lakes, and meltwater channels. Multi-modal fusion reduces confusion between snow shadows and dark water.
5.2 Amazon Basin River Network Mapping (TensorFlow): U-Net segmentation on Sentinel-2 mosaics with MNDWI thresholding as weak labels. Post-process predictions with morphological operations to produce connected river graphs for navigation and planning.
5.3 Urban Reservoir and Wetland Identification (Multispectral Fusion): Combine SWIR with NIR to reduce rooftop/asphalt false positives. Employ object-based filtering to remove small dark features that are unlikely to be water (e.g., parking lots).

   
6. Deployment and Scalability -
- Batch inference on cloud GPUs (Azure, AWS, GCP) over tiled datasets stored in cloud object storage (e.g., Cloud-Optimized GeoTIFFs).
- Use sliding-window inference with overlap to minimize edge artifacts; stitch predictions using weighted blending.
- Integrate outputs into GIS systems (GeoPackage, PostGIS) for visualization, querying, and downstream decision-making.
- Build alerting pipelines that flag new or expanding water bodies from time-series analysis (e.g., flooding, seasonal wetlands).

  
7. Future Directions -
- Incorporate SAR (Sentinel-1) more systematically for cloud-prone regions; explore polarimetric features and coherence for water surface smoothness.
- Multi-modal learning: fuse optical + SAR + DEM (topography) to prioritize likely clean sources (elevational springs, glacial lakes) and filter out urban runoff.
- Uncertainty quantification (MC Dropout or ensembles) to produce confidence maps that aid field validation and resource allocation.
- Active learning with human-in-the-loop labeling to refine models in novel geographies.

  
8. Limitations and Ethical Considerations -
- Water detected via remote sensing reflects surface presence and spectral signatures; it does not guarantee potability. Field validation remains essential.
- Seasonal and atmospheric variability (haze, sun glint) can affect detection; radar helps but introduces its own complexities.
- Publishing precise locations of pristine water sources may have ecological and social implications; consider access control and responsible disclosure.

<img width="1536" height="1024" alt="Designer (1)" src="https://github.com/user-attachments/assets/aff17e95-a41c-4b63-b801-61230438842f" />


9. Conclusion -
TensorFlow and PyTorch provide robust, flexible tooling for large-scale geospatial analysis. Using segmentation architectures (U-Net, DeepLab variants), multispectral inputs, and thoughtful preprocessing, practitioners can accurately delineate natural water sources across diverse terrains. The outlined pipelines and examples demonstrate how to operationalize these methods for monitoring, planning, and sustainability initiatives.

Credits: CoPilot 

[[[[Notes:
Calculate NDWI for the new Udupi data
The NDWI for the Udupi data has already been calculated, and the water bodies detected. I've also generated an interactive map visualizing these. Here's a summary of the visualization and the steps taken:

        "\n",

Explain how NDWI was calculated for Udupi data
Certainly! For the Udupi data, the Normalized Difference Water Index (NDWI) was calculated using the following formula, which is standard for Sentinel-2 imagery:

NDWI = (Green - NIR) / (Green + NIR)
Specifically, in the code, this was implemented as:

The Green band was represented by Sentinel-2 Band B3.
The Near-Infrared (NIR) band was represented by Sentinel-2 Band B8.
So, for each pixel in the preprocessed Sentinel-2 image for Udupi, the Earth Engine's normalizedDifference function was applied to bands B3 and B8 to compute the NDWI value. This resulted in a new band named 'NDWI' being added to the image, with values typically ranging from -1 to 1. Higher NDWI values generally indicate the presence of water.

        "\n",

what does temporal change indicate? how is it relevant to clean water attributes?
Temporal change, in the context of water bodies, refers to how their extent, shape, or characteristics evolve over different periods. This could be seasonal changes, year-over-year variations, or longer-term trends.

Its relevance to clean water attributes, while not directly measured by simple water body detection, is significant:

Availability: A primary attribute of 'clean water sources' is the sheer availability of water. If water bodies are shrinking or disappearing over time (negative temporal change), it directly impacts the availability of any water, clean or otherwise.
Concentration of Pollutants: When water bodies shrink due to factors like drought or excessive extraction, the remaining water can have a higher concentration of pollutants, dissolved solids, or sediments. This decrease in volume can lead to a degradation of water quality.
Hydrological Stress: Significant temporal fluctuations, particularly losses, can indicate hydrological stress in a region, which often correlates with compromised water quality due to reduced dilution capacity or increased human intervention (e.g., pumping dry wells).
Ecosystem Health: Stable and healthy water bodies are often indicators of balanced ecosystems, which are crucial for maintaining water quality. Drastic temporal changes can disrupt these ecosystems, leading to algal blooms, increased turbidity, or other water quality issues.
Sedimentation and Turbidity: Changes in water levels can lead to increased erosion and sedimentation, especially during periods of water loss and subsequent refilling, directly impacting turbidity (a key clean water attribute) and potentially carrying pollutants.
While this temporal analysis primarily focused on quantifying the change in presence of water, understanding these changes is a fundamental first step. If water bodies are stable and persistent, it provides a baseline for more advanced water quality assessments. If they are dynamic or shrinking, it signals potential areas where water quality might be under threat and warrants further, more detailed investigation using specific water quality indices or in-situ measurements.


algal blooms, increased turbidity, or other water quality issues - can they be detected in some form via satellite images, if yes which of the satellites have such data?
Yes, several water quality issues, including algal blooms and increased turbidity, can be detected or inferred from satellite imagery. This is primarily done by analyzing the spectral reflectance of water in different wavelength bands, as various constituents in the water absorb and scatter light differently.

Here's how they are generally detected and which satellites are suitable:
        "\n",

Algal Blooms (Chlorophyll-a Concentration):

Detection Principle: Chlorophyll-a, the primary pigment in algae, absorbs strongly in the blue and red parts of the visible spectrum and reflects strongly in the green and near-infrared (NIR) regions. Algal blooms cause a characteristic 'green-up' in the water's spectral signature, or in severe cases, red tides. Specialized indices (like various chlorophyll indices) can be calculated to quantify chlorophyll-a.
Suitable Satellites:
Sentinel-2 (MSI): With its visible and red-edge bands, it's excellent for detecting and mapping chlorophyll-a concentrations in inland and coastal waters at high spatial resolution (10-20m).
Sentinel-3 (OLCI): Specifically designed for ocean and inland water color monitoring, OLCI has many narrow spectral bands optimized for chlorophyll-a detection, making it very sensitive to subtle changes.
Landsat (OLI/OLI-2): Offers long-term data for historical analysis of chlorophyll-a, using its visible and NIR bands, though with broader spectral bands than Sentinel-2/3.
MODIS (Aqua/Terra): Provides daily global coverage, useful for tracking large-scale algal blooms in larger lakes and oceans, despite its coarser spatial resolution.
Increased Turbidity (Total Suspended Matter - TSM):

Detection Principle: Turbidity is caused by suspended particles (sediments, organic matter, plankton) that increase light scattering in the water. This typically leads to higher reflectance across the visible and NIR spectrum. The red and NIR bands are often most sensitive to variations in TSM.
Suitable Satellites:
Sentinel-2 (MSI): Its visible and NIR bands are effective for measuring and mapping turbidity variations at high spatial resolution.
Landsat (OLI/OLI-2): Similar to chlorophyll-a, Landsat's bands are suitable for turbidity monitoring and historical trend analysis.
Sentinel-3 (OLCI): Also well-suited for TSM mapping due to its optimized spectral bands for water quality.
MODIS (Aqua/Terra): Good for large-scale turbidity monitoring due to its high temporal resolution.
Other Water Quality Issues (e.g., Colored Dissolved Organic Matter (CDOM), Water Temperature):

CDOM: Absorbs strongly in the blue and green light. Can be detected with visible bands from satellites like Sentinel-2, Landsat, and Sentinel-3 (especially OLCI).
Water Temperature: Thermal infrared sensors can detect surface water temperature. Anomalies might indicate thermal pollution or upwelling. Satellites like Landsat (TIRS) and MODIS (Thermal Infrared bands) provide this data.
In summary, modern Earth observation satellites like Sentinel-2 and Sentinel-3 are particularly powerful for water quality monitoring due to their optimized spectral bands and repeat visit times, while Landsat provides valuable historical context, and MODIS is excellent for large-scale, frequent observations.
        "\n",


1. Each Sentinel-2 satellite takes 100.6 minutes for an entire orbit, with S2A and S2B orbiting 180 degree apart. By setting a time interval to less than 50 minutes, there should be no overlap possible between different acquisitions, even near the poles. How is it we could not get data for 2025 for our first AOI? 2. For atmospheric correction, ESA's official Sen2Cor is used. In most cases the processing is done by ESA itself. Are cloud and snow masks applied by the onboard sw itself?
Let's break down your questions about Sentinel-2:

1. Why no data for 2025 for our first AOI?

The reason we couldn't get data for 2025 for your first Area of Interest (AOI) is very straightforward: 2025 is in the future.

Satellite data, including Sentinel-2 imagery, is collected in real-time as the satellites orbit Earth. Google Earth Engine (and similar platforms) provide access to archives of previously collected satellite imagery. Since we were trying to query for imagery in January 2025, that data had not yet been acquired by the Sentinel-2 satellites. The orbit mechanics and overlap times you mentioned refer to how frequently an area can be observed, but not to the existence of data for future periods.

2. Are cloud and snow masks applied by the onboard software itself?

No, cloud and snow masks are not applied by the onboard software of the Sentinel-2 satellites. They are generated as part of the ground processing segment.

Here's a brief overview of the Sentinel-2 processing levels:

Level-0 (Raw Data): This is the raw telemetry data transmitted from the satellite.
Level-1A/B (Radiometric Correction): Initial processing for radiometric calibration.
Level-1C (Top-of-Atmosphere Reflectance): This product is generated on the ground and provides atmospherically corrected data at the top of the atmosphere. While it's corrected, it still contains atmospheric effects like clouds and aerosols.
Level-2A (Bottom-of-Atmosphere Reflectance): This is the product we typically use (e.g., COPERNICUS/S2_SR_HARMONIZED). It is derived from Level-1C data using atmospheric correction processors like ESA's Sen2Cor. It's during this Level-2A processing that the Scene Classification Layer (SCL), which includes information about clouds, cloud shadows, and snow/ice, as well as quality indicators like QA60, are generated and applied. These masks are crucial for filtering out unwanted atmospheric interference from the imagery.
So, the onboard software captures the raw data, but the sophisticated algorithms for atmospheric correction and scene classification (including cloud/snow masking) happen on powerful ground-based systems.



To understand the nuances of these two powerful satellite constellations, Here's how you can separate their data and their respective strengths for different types of analysis:

Separating Landsat and Sentinel-2 Data
In Google Earth Engine, you typically work with specific Image Collections. The easiest way to separate Landsat and Sentinel-2 data is by simply querying their respective collection IDs:

For Landsat: You'll generally use collections like 'LANDSAT/LC08/C01/T1_SR' (Landsat 8 Surface Reflectance, Collection 1) or 'LANDSAT/LC09/C02/T1_SR' (Landsat 9 Surface Reflectance, Collection 2). The collection ID itself distinguishes it as Landsat data.
For Sentinel-2: You'll use collections like 'COPERNICUS/S2_SR_HARMONIZED' (Sentinel-2 Level-2A Surface Reflectance).
If you happen to combine them into a single collection (though less common for direct comparison unless you're mapping them to a common schema), you can also filter by metadata properties like PLATFORM_NAME or SPACECRAFT_ID. For example:

# Filter for Landsat 8
landsat8_collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')\
    .filter(ee.Filter.eq('PLATFORM_NAME', 'LANDSAT_8'))

# Filter for Sentinel-2
sentinel2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
    .filter(ee.Filter.eq('PLATFORM_NAME', 'Sentinel-2'))
Which for Thermal/Spectral Analysis?
This is where their capabilities diverge significantly:

Thermal Analysis (Surface Temperature):

Landsat is the primary choice for thermal analysis. Landsat satellites (specifically Landsat 8 and 9 with the Thermal Infrared Sensor - TIRS) have dedicated thermal bands. These bands are crucial for retrieving Land Surface Temperature (LST), which is vital for studying urban heat islands, evapotranspiration, water temperature, and geological features.
Sentinel-2 does NOT have thermal bands. Therefore, it cannot be used for direct thermal analysis.
Spectral Analysis (Vegetation, Water Quality, Land Cover):

Both Landsat and Sentinel-2 are excellent for spectral analysis, but with different strengths.
Sentinel-2: Offers higher spatial resolution (10m for visible and NIR bands, 20m for red-edge and SWIR) and more spectral bands, particularly the red-edge bands, which are very useful for vegetation health monitoring, precise land cover classification, and some water quality parameters like chlorophyll detection. Its higher revisit frequency (5 days with two satellites) also provides more frequent observations.
Landsat: Provides a longer historical archive (dating back to the 1970s), which is invaluable for long-term change detection. Its bands (visible, NIR, SWIR) are suitable for a wide range of spectral analyses, including NDWI (as we used), NDVI, and various land cover mapping applications, though at a coarser spatial resolution (30m) compared to Sentinel-2's core bands.
In summary:

For thermal analysis, Landsat is your go-to.
For spectral analysis, both are valuable. Choose Sentinel-2 for higher spatial detail, more spectral richness (red-edge), and higher temporal frequency. Choose Landsat for historical trend analysis and a consistent long-term record."]]]]

11. References -
- Sentinel-2 MSI User Guide; Landsat 8/9 OLI Instrument Guides
- McFeeters, S.K. (1996). The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features.
- Xu, H. (2006). Modification of normalised difference water index (NDWI) to enhance open water features: A comparison of Landsat imagery.
- TorchGeo documentation; TensorFlow/Keras documentation; torchvision models and segmentation references
