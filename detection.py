import os
import cv2
import argparse
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm import tqdm
from matplotlib import colors
from shapely.geometry import Point



def detect_at_multiple_resolutions(image_path, model, output_dir, tile_size=640, overlap=32,
                                   resolutions=[1.0, 0.5, 0.3], conf_threshold=0.25):
    """
    Multi-resolution detection: downscale image and detect at each resolution.
    Returns detection points, bounding boxes, and tile stats.
    """

    all_points = []
    detection_points_pixel = []
    detection_boxes = []
    counts_per_tile = []

    fig, axes = plt.subplots(1, len(resolutions), figsize=(6*len(resolutions), 12))
    if len(resolutions) == 1:
        axes = [axes]

    cmap = plt.cm.plasma
    norm = colors.Normalize(vmin=0, vmax=20)

    with rasterio.open(image_path) as src:
        original_width, original_height = src.width, src.height
        transform = src.transform

        print(f"Original image size: {original_width}x{original_height}")

        for res_idx, resolution in enumerate(resolutions):
            print(f"\n--- Processing at {resolution}x resolution ---")

            new_width = int(original_width * resolution)
            new_height = int(original_height * resolution)
            print(f"Scaled image size: {new_width}x{new_height}")

            full_image = src.read([1, 2, 3])
            full_image = np.transpose(full_image, (1, 2, 0))
            full_image = np.clip(full_image, 0, 255).astype(np.uint8)

            resized_image = cv2.resize(full_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            background_img = resized_image.astype(np.float32) / 255.0

            ax = axes[res_idx]
            ax.set_title(f"Resolution {resolution}x ({new_width}x{new_height})", fontsize=12)
            ax.imshow(background_img, extent=[0, new_width, new_height, 0], alpha=0.7)
            ax.set_xlim(0, new_width)
            ax.set_ylim(new_height, 0)
            ax.set_axis_off()

            resolution_detections = []
            resolution_counts = []
            tile_id = 0

            for y in tqdm(range(0, new_height, tile_size - overlap), desc=f"Resolution {resolution}x"):
                for x in range(0, new_width, tile_size - overlap):
                    x_end = min(x + tile_size, new_width)
                    y_end = min(y + tile_size, new_height)
                    win_w = x_end - x
                    win_h = y_end - y

                    if win_w < tile_size * 0.5 or win_h < tile_size * 0.5:
                        continue

                    tile = resized_image[y:y_end, x:x_end]
                    if tile.shape[:2] != (tile_size, tile_size):
                        tile = cv2.resize(tile, (tile_size, tile_size))

                    tile = np.ascontiguousarray(tile)
                    results = model.predict(source=tile, conf=conf_threshold,
                                            save_txt=False, save_conf=True, verbose=False)

                    detection_count = 0
                    if results[0].boxes is not None and len(results[0].boxes) > 0:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        confidences = results[0].boxes.conf.cpu().numpy()
                        detection_count = len(boxes)

                        for i, bx in enumerate(boxes):
                            if tile.shape[:2] != (tile_size, tile_size):
                                scale_x = win_w / tile_size
                                scale_y = win_h / tile_size
                                bx_scaled = bx * np.array([scale_x, scale_y, scale_x, scale_y])
                            else:
                                bx_scaled = bx

                            cx = (bx_scaled[0] + bx_scaled[2]) / 2
                            cy = (bx_scaled[1] + bx_scaled[3]) / 2

                            global_x = x + cx
                            global_y = y + cy

                            original_x = global_x / resolution
                            original_y = global_y / resolution

                            px, py = rasterio.transform.xy(transform, original_y, original_x)

                            detection_data = {
                                'x': original_x,
                                'y': original_y,
                                'confidence': confidences[i],
                                'resolution': resolution,
                                'tile_id': tile_id
                            }

                            resolution_detections.append(detection_data)
                            all_points.append(Point(px, py))

                            ax.scatter(global_x, global_y, c='red', s=30, alpha=0.8,
                                       edgecolors='white', linewidths=1)

                            detection_boxes.append({
                                'x1': original_x - (bx_scaled[2] - bx_scaled[0])/(2*resolution),
                                'y1': original_y - (bx_scaled[3] - bx_scaled[1])/(2*resolution),
                                'x2': original_x + (bx_scaled[2] - bx_scaled[0])/(2*resolution),
                                'y2': original_y + (bx_scaled[3] - bx_scaled[1])/(2*resolution),
                                'confidence': confidences[i],
                                'resolution': resolution,
                                'tile_id': tile_id
                            })

                    resolution_counts.append({
                        "tile_id": tile_id,
                        "count": detection_count,
                        "resolution": resolution,
                        "x": x, "y": y,
                        "width": win_w, "height": win_h
                    })

                    if detection_count > 0:
                        rect = patches.Rectangle((x, y), win_w, win_h,
                                                 linewidth=1, edgecolor=cmap(norm(detection_count)),
                                                 facecolor=cmap(norm(detection_count)), alpha=0.2)
                        ax.add_patch(rect)
                        ax.text(x + 10, y + 30, str(detection_count),
                                color='white', fontsize=8, weight='bold',
                                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.8))

                    tile_id += 1

            total_det = len(resolution_detections)
            avg_conf = np.mean([d['confidence'] for d in resolution_detections]) if total_det > 0 else 0

            stats_text = f"Detections: {total_det}\nAvg Conf: {avg_conf:.3f}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle="round", fc="white", alpha=0.8))

            detection_points_pixel.extend(resolution_detections)
            counts_per_tile.extend(resolution_counts)

            print(f"Resolution {resolution}x: {len(resolution_detections)} detections")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/multi_resolution_detection.png', dpi=300, bbox_inches='tight')
    plt.show()

    return all_points, detection_points_pixel, detection_boxes, counts_per_tile


def main():
    parser = argparse.ArgumentParser(description="Multi-resolution object detection with YOLO")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to YOLO model file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--conf_threshold", type=float, default=0.25, help="Confidence threshold for detection")
    args = parser.parse_args()

    tile_size = 640
    overlap = 32
    bands = [1, 2, 3]

    print("Starting multi-resolution tree detection...")
    resolutions = [1.0, 0.75, 0.5, 0.3, 0.25]
    print(f"Resolutions: {', '.join([str(r)+'x' for r in resolutions])}")

    from ultralytics import YOLO
    new_model = YOLO(args.model_path)

    all_points, detection_points_pixel, detection_boxes, counts_per_tile = detect_at_multiple_resolutions(
        image_path=args.image_path,
        model=new_model,
        output_dir=args.output_dir,
        tile_size=tile_size,
        overlap=overlap,
        resolutions=resolutions,
        conf_threshold=args.conf_threshold
    )

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_title("Combined Multi-Resolution Detections", fontsize=16)
    ax.set_axis_off()

    with rasterio.open(args.image_path) as src:
        width, height = src.width, src.height
        background_img = src.read(bands)
        background_img = np.transpose(background_img, (1, 2, 0))
        background_img = np.clip(background_img / 255.0, 0, 1)

    ax.imshow(background_img, extent=[0, width, height, 0], alpha=0.7)

    if detection_points_pixel:
        resolutions = list(set([d['resolution'] for d in detection_points_pixel]))
        colors_res = ['red', 'blue', 'green', 'orange', 'purple']

        for i, res in enumerate(sorted(resolutions, reverse=True)):
            res_detections = [d for d in detection_points_pixel if d['resolution'] == res]
            if res_detections:
                x_coords = [d['x'] for d in res_detections]
                y_coords = [d['y'] for d in res_detections]
                ax.scatter(x_coords, y_coords, c=colors_res[i % len(colors_res)],
                           s=50, alpha=0.7, label=f'{res}x resolution ({len(res_detections)})',
                           edgecolors='white', linewidths=1)

        ax.legend(loc='upper right', fontsize=12)

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    plt.savefig(f'{args.output_dir}/combined_multi_resolution.png', dpi=300, bbox_inches='tight')
    plt.show()

    if detection_points_pixel:
        df_detections = pd.DataFrame(detection_points_pixel)
        df_detections.to_csv(f'{args.output_dir}/multi_resolution_detections.csv', index=False)

        if all_points:
            gdf_points = gpd.GeoDataFrame(geometry=all_points)
            gdf_points['confidence'] = [d['confidence'] for d in detection_points_pixel]
            gdf_points['resolution'] = [d['resolution'] for d in detection_points_pixel]
            gdf_points['tile_id'] = [d['tile_id'] for d in detection_points_pixel]

            try:
                gdf_points.to_file(f"{args.output_dir}/multi_resolution_points.geojson", driver="GeoJSON")
            except Exception as e:
                print(f"Warning: Could not save GeoJSON ({e})")

        if detection_boxes:
            df_boxes = pd.DataFrame(detection_boxes)
            df_boxes.to_csv(f'{args.output_dir}/multi_resolution_boxes.csv', index=False)

        df_counts = pd.DataFrame(counts_per_tile)
        df_counts.to_csv(f'{args.output_dir}/multi_resolution_tile_counts.csv', index=False)

    print(f"\n=== MULTI-RESOLUTION DETECTION SUMMARY ===")
    if detection_points_pixel:
        by_resolution = {}
        for det in detection_points_pixel:
            res = det['resolution']
            if res not in by_resolution:
                by_resolution[res] = []
            by_resolution[res].append(det['confidence'])

        for res in sorted(by_resolution.keys(), reverse=True):
            detections = by_resolution[res]
            print(f"Resolution {res}x: {len(detections)} detections, avg confidence: {np.mean(detections):.3f}")

        print(f"\nTotal unique detections: {len(detection_points_pixel)}")
        print(f"Overall average confidence: {np.mean([d['confidence'] for d in detection_points_pixel]):.3f}")

    print(f"\nSaved files:")
    print(f"- multi_resolution_detections.csv")
    print(f"- multi_resolution_points.geojson")
    print(f"- multi_resolution_boxes.csv")
    print(f"- multi_resolution_tile_counts.csv")
    print(f"- multi_resolution_detection.png")
    print(f"- combined_multi_resolution.png")


if __name__ == "__main__":
    main()