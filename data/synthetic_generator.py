import os
import torch
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter, shift as ndimage_shift

class SyntheticSatelliteGenerator:
    """
    Generates realistic multi-sensor and multi-angle synthetic satellite scenes with matching
    ground truth land-cover semantic segmentation masks.
    
    Sensor domains simulated:
      - 'sentinel2'      : Multispectral (RGB + NIR), clean reflectance profiles
      - 'landsat8'       : Multispectral (RGB + NIR), warmer calibration shift, slight haze
      - 'spot6'          : High-contrast optical, sharp boundaries, texture variance
      - 'sentinel1_sar'  : Radar backscatter with speckle noise and distinct radar physics

    Angle domains simulated (off-nadir geometric distortions):
      - 'nadir_0'        : Perfectly vertical acquisition, no distortion (reference)
      - 'offnadir_15'    : Mild off-nadir (subtle lean + slight blur)
      - 'offnadir_30'    : Moderate off-nadir (visible building lean + foreshortening)
      - 'oblique_45'     : Severe oblique view (heavy building layover + significant blur)
    """
    
    # Maps angle-domain name -> acquisition angle in degrees
    ANGLE_DEGREES = {
        "nadir_0": 0,
        "offnadir_15": 15,
        "offnadir_30": 30,
        "oblique_45": 45,
    }
    
    def __init__(self, image_size=128, seed=42):
        self.image_size = image_size
        self.rng = np.random.RandomState(seed)
        
        # Classes: 0: Background, 1: Forest/Vegetation, 2: Water, 3: Urban/Buildings, 4: Agriculture, 5: Bare Soil
        self.class_names = ["Background", "Vegetation", "Water", "Urban", "Agriculture", "BareSoil"]
        
        # Base spectral signatures [R, G, B, NIR] in [0, 1]
        self.base_signatures = {
            0: [0.30, 0.30, 0.30, 0.30], # Background
            1: [0.10, 0.35, 0.12, 0.85], # Vegetation (Low Red, High NIR)
            2: [0.05, 0.15, 0.45, 0.03], # Water (Strong Blue, very low NIR)
            3: [0.55, 0.55, 0.58, 0.50], # Urban (High Reflectance across all)
            4: [0.25, 0.50, 0.20, 0.70], # Agriculture (Moderate green + high NIR)
            5: [0.65, 0.45, 0.30, 0.40], # Bare Soil (High Red/Green, moderate NIR)
        }

    def _generate_terrain_mask(self):
        """Generates a cohesive geospatial terrain map with continuous regions"""
        H, W = self.image_size, self.image_size
        x = np.linspace(-3, 3, W)
        y = np.linspace(-3, 3, H)
        xx, yy = np.meshgrid(x, y)
        
        # Multi-frequency smooth terrain potential
        f1 = np.sin(xx * 1.5 + self.rng.uniform(-np.pi, np.pi)) + np.cos(yy * 1.5 + self.rng.uniform(-np.pi, np.pi))
        f2 = 0.5 * np.sin(xx * 3.0 + yy * 2.0 + self.rng.uniform(-np.pi, np.pi))
        noise = 0.15 * self.rng.randn(H, W)
        potential = f1 + f2 + noise
        
        mask = np.zeros((H, W), dtype=np.int64)
        
        # River / Water body
        river_pos = np.sin(xx * 0.8 + self.rng.uniform(-1, 1)) * 1.2
        water_mask = np.abs(yy - river_pos) < 0.4
        mask[water_mask] = 2
        
        # Assign other land cover types based on terrain potential
        non_water = ~water_mask
        p_nw = potential[non_water]
        
        # Quantile thresholding for natural spatial clustering
        q20, q50, q75, q90 = np.percentile(p_nw, [20, 50, 75, 90])
        
        mask[non_water & (potential < q20)] = 1  # Forest/Vegetation
        mask[non_water & (potential >= q20) & (potential < q50)] = 4  # Agriculture
        mask[non_water & (potential >= q50) & (potential < q75)] = 5  # Bare Soil
        mask[non_water & (potential >= q75) & (potential < q90)] = 3  # Urban
        mask[non_water & (potential >= q90)] = 0  # Background / Mixed
        
        return mask

    def render_sensor_image(self, mask: np.ndarray, domain: str):
        """
        Renders multi-channel sensor imagery from ground truth mask according to sensor domain physics.
        Output: (4, H, W) float32 tensor normalized in [-1, 1]
        """
        H, W = mask.shape
        img = np.zeros((4, H, W), dtype=np.float32)
        
        # Fill base spectral signatures
        for cls_idx, sig in self.base_signatures.items():
            cls_mask = (mask == cls_idx)
            if not np.any(cls_mask):
                continue
            for c in range(4):
                img[c, cls_mask] = sig[c]
                
        # Add spatial texture noise per landcover
        spatial_noise = self.rng.normal(0, 0.04, (4, H, W)).astype(np.float32)
        img += spatial_noise

        if domain == "sentinel2":
            # Canonical standard optical-NIR
            # Slight atmospheric haze
            img[2] += 0.03 # Blue scatter
            img = np.clip(img, 0.0, 1.0)
            
        elif domain == "landsat8":
            # Warmer radiometric calibration & higher atmospheric attenuation
            gain = np.array([1.10, 0.98, 0.90, 1.05], dtype=np.float32).reshape(4, 1, 1)
            bias = np.array([0.05, 0.03, 0.01, 0.04], dtype=np.float32).reshape(4, 1, 1)
            img = np.clip(img * gain + bias, 0.0, 1.0)
            
        elif domain == "spot6":
            # Very High Resolution optical: higher contrast, sharp spatial frequency, cooler tone
            mean = img.mean(axis=(1, 2), keepdims=True)
            img = (img - mean) * 1.3 + mean
            img[0] *= 0.95 # Less red
            img[2] *= 1.08 # Crisp blue
            img = np.clip(img, 0.0, 1.0)
            
        elif domain == "sentinel1_sar":
            # Radar physics simulation:
            # Band 0: VV polarization (Backscatter intensity)
            # Band 1: VH polarization (Cross polarization)
            # Band 2: VV/VH ratio
            # Band 3: Radar Texture / Coherence
            
            sar_vv = np.zeros((H, W), dtype=np.float32)
            sar_vh = np.zeros((H, W), dtype=np.float32)
            
            # Urban: strong double bounce (high backscatter)
            sar_vv[mask == 3] = 0.85
            sar_vh[mask == 3] = 0.70
            
            # Forest/Vegetation: volumetric diffuse scattering
            sar_vv[mask == 1] = 0.45
            sar_vh[mask == 1] = 0.40
            
            # Agriculture: moderate surface + volume scatter
            sar_vv[mask == 4] = 0.35
            sar_vh[mask == 4] = 0.25
            
            # Bare Soil: rough surface scatter
            sar_vv[mask == 5] = 0.30
            sar_vh[mask == 5] = 0.15
            
            # Water: specular reflection (almost black in radar)
            sar_vv[mask == 2] = 0.03
            sar_vh[mask == 2] = 0.01
            
            sar_vv[mask == 0] = 0.20
            sar_vh[mask == 0] = 0.10
            
            # Multiplicative Rayleigh/Gamma speckle noise
            speckle = self.rng.gamma(shape=3.0, scale=1.0/3.0, size=(2, H, W)).astype(np.float32)
            sar_vv = sar_vv * speckle[0]
            sar_vh = sar_vh * speckle[1]
            
            ratio = (sar_vv + 1e-4) / (sar_vh + 1e-4)
            ratio = np.clip(ratio / 5.0, 0.0, 1.0)
            texture = np.clip((sar_vv - sar_vh) * 1.5 + 0.3, 0.0, 1.0)
            
            img[0] = np.clip(sar_vv, 0.0, 1.0)
            img[1] = np.clip(sar_vh, 0.0, 1.0)
            img[2] = ratio
            img[3] = texture
            
        else:
            img = np.clip(img, 0.0, 1.0)
            
        # Scale to [-1, 1] for GAN stability
        norm_img = (img * 2.0) - 1.0
        return torch.from_numpy(norm_img.astype(np.float32))

    def render_angle_image(self, mask: np.ndarray, domain: str):
        """
        Renders multi-channel imagery from a nadir-view ground truth mask, then applies
        physically-motivated geometric distortions for the given off-nadir acquisition angle.

        Distortions applied (proportional to tan/cos of angle):
          1. Radiometric brightness fall-off: cos(θ) attenuation across all bands.
          2. GSD blur: Effective resolution degrades at higher angles — simulated via
             a uniform spatial blur whose kernel width scales with tan(θ).
          3. Building lean / layover: Urban pixels displaced in X by tan(θ) * height_proxy.
          4. Foreshortening: Horizontal pixel compression on the far range side.

        Output: (4, H, W) float32 tensor normalized in [-1, 1]
        """
        angle_deg = self.ANGLE_DEGREES.get(domain, 0)
        theta = np.radians(angle_deg)

        # Start from nadir-view sentinel2 rendering as the "truth" base
        img = np.zeros((4, mask.shape[0], mask.shape[1]), dtype=np.float32)
        for cls_idx, sig in self.base_signatures.items():
            cls_mask = (mask == cls_idx)
            if not np.any(cls_mask):
                continue
            for c in range(4):
                img[c, cls_mask] = sig[c]

        # Spatial texture noise
        img += self.rng.normal(0, 0.04, img.shape).astype(np.float32)
        img = np.clip(img, 0.0, 1.0)

        if angle_deg == 0:
            # Pure nadir — no distortion needed
            norm_img = (img * 2.0) - 1.0
            return torch.from_numpy(norm_img.astype(np.float32))

        H, W = mask.shape

        # 1. Radiometric brightness fall-off: at higher angles, less energy per pixel
        brightness_scale = np.cos(theta)   # cos(0°)=1.0, cos(45°)≈0.707
        img *= brightness_scale

        # 2. GSD blur: resolution degrades with angle; kernel proportional to tan(θ)
        blur_sigma = max(1.0, np.tan(theta) * 4.0)  # 0 at nadir, ~4 at 45°
        for c in range(4):
            img[c] = uniform_filter(img[c], size=int(blur_sigma))

        # 3. Building lean (layover): shift Urban class pixels along X-axis
        #    Displacement = tan(θ) * height_proxy (arbitrary units scaled to image)
        lean_pixels = int(np.tan(theta) * H * 0.08)  # max ~8% image width at 45°
        if lean_pixels > 0:
            urban_mask = (mask == 3)  # Urban
            for c in range(4):
                channel = img[c].copy()
                # Shift urban pixels right (simulating building tops leaning away)
                leaned = ndimage_shift(channel * urban_mask, shift=[0, lean_pixels], mode="nearest")
                # Blend lean into image: urban base stays, top is shifted
                non_lean = channel * (~urban_mask).astype(np.float32)
                img[c] = non_lean + leaned

        # 4. Foreshortening: compress image horizontally by cos(θ) then pad back
        compressed_W = max(1, int(W * np.cos(theta)))
        if compressed_W < W:
            for c in range(4):
                compressed = np.array(
                    Image.fromarray(img[c]).resize((compressed_W, H), Image.BILINEAR)
                )
                # Pad right side with edge values
                padded = np.pad(compressed, ((0, 0), (0, W - compressed_W)), mode="edge")
                img[c] = padded

        img = np.clip(img, 0.0, 1.0)
        norm_img = (img * 2.0) - 1.0
        return torch.from_numpy(norm_img.astype(np.float32))

    def generate_dataset(self, output_dir: str, num_samples_per_domain: int = 100, domains=None):
        """Pre-generates and caches dataset files to disk.
        
        Supports both sensor domains (e.g. 'sentinel2') and angle domains (e.g. 'offnadir_30').
        Angle domains are rendered from the same terrain masks using geometric distortion simulation.
        """
        if domains is None:
            domains = ["sentinel2", "landsat8", "spot6", "sentinel1_sar"]
            
        for domain in domains:
            domain_dir = os.path.join(output_dir, domain)
            os.makedirs(os.path.join(domain_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(domain_dir, "masks"), exist_ok=True)
            
            is_angle_domain = domain in self.ANGLE_DEGREES
            domain_type = "angle" if is_angle_domain else "sensor"
            print(f"[Dataset Generator] Generating {num_samples_per_domain} patches for {domain_type} domain: {domain}...")
            
            for i in range(num_samples_per_domain):
                mask = self._generate_terrain_mask()
                if is_angle_domain:
                    img_tensor = self.render_angle_image(mask, domain)
                else:
                    img_tensor = self.render_sensor_image(mask, domain)
                
                # Save tensors
                sample_name = f"patch_{i:04d}.pt"
                torch.save(img_tensor, os.path.join(domain_dir, "images", sample_name))
                torch.save(torch.from_numpy(mask), os.path.join(domain_dir, "masks", sample_name))
                
        print(f"[Dataset Generator] Successfully generated benchmark data across all domains at {output_dir}")
