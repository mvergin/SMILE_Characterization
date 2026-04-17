import sys
import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_large_font(target_size=72):
    """Safely loads a large font. Falls back through OS defaults if Arial isn't found."""
    try:
        # Pillow >= 10.1.0 supports the size parameter on the default font
        return ImageFont.load_default(size=target_size)
    except Exception:
        pass

    # Common TrueType fonts across Windows, Linux, and macOS
    fonts = [
        "arial.ttf",
        "Arial.ttf",
        "calibri.ttf",
        "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf",
        "sf-pro-display-regular.otf",
        "Helvetica.ttc",
    ]
    for f in fonts:
        try:
            return ImageFont.truetype(f, target_size)
        except Exception:
            continue

    # Absolute fallback (will be small, but prevents crashing)
    return ImageFont.load_default()


def interpolate_color(value, min_val, max_val):
    """Maps a value to a Red (min) -> Yellow -> Green (max) gradient."""
    if np.isnan(value):
        return (0, 0, 0)
    if max_val == min_val:
        return (0, 255, 0)

    ratio = (value - min_val) / (max_val - min_val)
    if ratio < 0.5:
        sub_ratio = ratio * 2
        return (255, int(255 * sub_ratio), 0)
    else:
        sub_ratio = (ratio - 0.5) * 2
        return (int(255 * (1 - sub_ratio)), 255, 0)


def format_value(val):
    """Format value to clean SI prefix units representing integers."""
    if val == 0:
        return "0"
    abs_val = abs(val)
    if abs_val >= 1:
        return f"{int(round(val))}"
    elif abs_val >= 1e-3:
        return f"{int(round(val * 1e3))}m"
    elif abs_val >= 1e-6:
        return f"{int(round(val * 1e6))}u"
    elif abs_val >= 1e-9:
        return f"{int(round(val * 1e9))}n"
    elif abs_val >= 1e-12:
        return f"{int(round(val * 1e12))}p"
    else:
        return f"{val:.1e}"


def generate_images(csv_path, output_dir=None):
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return

    print(f"Processing {csv_path}...")

    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_dir = os.path.join(os.path.dirname(csv_path), base_name + "_images")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    req_cols = ["X", "Y", "BITVAL", "TYPE", "MEAS_VALUE"]
    if not all(col in df.columns for col in req_cols):
        print(f"Error: CSV missing columns. required: {req_cols}")
        return

    has_nvled_v = "NVLED_V" in df.columns

    # 1. Calculate dynamic Grid boundaries
    min_x, max_x = int(df["X"].min()), int(df["X"].max())
    min_y, max_y = int(df["Y"].min()), int(df["Y"].max())

    grid_w = max_x - min_x + 1
    grid_h = max_y - min_y + 1
    print(
        f"Detected actual bounding box: {grid_w}x{grid_h} (X: {min_x}-{max_x}, Y: {min_y}-{max_y})"
    )

    if grid_w == 512 and grid_h == 512:
        area_label = "FULL"
    elif grid_w == 256 and grid_h == 256:
        area_label = "Quarter"
    else:
        area_label = f"ROI_{min_x}-{min_y}_to_{max_x}-{max_y}"

    # 2. Layout Settings
    BORDER_SIZE, INNER_SIZE = 1, 8
    BLOCK_SIZE = BORDER_SIZE + INNER_SIZE + BORDER_SIZE

    # Greatly widened to ensure size 72 text doesn't clip out of bounds
    COLORBAR_WIDTH = 400

    map_w = grid_w * BLOCK_SIZE
    map_h = grid_h * BLOCK_SIZE

    # Enforce a minimum height so a colorbar with large font ticks always has room
    img_height = max(map_h, 1100)
    total_width = map_w + COLORBAR_WIDTH

    # Center the map vertically if the required image height is taller than the map
    map_y_offset = (img_height - map_h) // 2

    font_size = 72
    font = get_large_font(font_size)

    print("Grouping and averaging data...")
    group_cols = ["X", "Y", "BITVAL", "TYPE"]
    if has_nvled_v:
        group_cols.append("NVLED_V")
    df_avg = df.groupby(group_cols)["MEAS_VALUE"].mean().reset_index()

    uniq_cols = ["BITVAL", "TYPE"]
    if has_nvled_v:
        uniq_cols.append("NVLED_V")
    unique_combinations = df_avg[uniq_cols].drop_duplicates()

    for _, row in unique_combinations.iterrows():
        bitval, dev_type = row["BITVAL"], row["TYPE"]
        nvled_v = row["NVLED_V"] if has_nvled_v else None

        cond = (df_avg["BITVAL"] == bitval) & (df_avg["TYPE"] == dev_type)
        if has_nvled_v:
            cond &= df_avg["NVLED_V"] == nvled_v
        subset = df_avg[cond]

        min_val, max_val = subset["MEAS_VALUE"].min(), subset["MEAS_VALUE"].max()

        img = Image.new("RGB", (total_width, img_height), "black")
        draw = ImageDraw.Draw(img)

        # Draw map pixels
        for _, px_row in subset.iterrows():
            x, y = int(px_row["X"]), int(px_row["Y"])
            color = interpolate_color(px_row["MEAS_VALUE"], min_val, max_val)

            # Map coordinates relative to the cropped boundary
            map_x = x - min_x
            map_y = y - min_y

            x0 = map_x * BLOCK_SIZE
            y0 = map_y * BLOCK_SIZE + map_y_offset

            draw.rectangle(
                [x0, y0, x0 + BLOCK_SIZE - 1, y0 + BLOCK_SIZE - 1], fill="white"
            )
            draw.rectangle(
                [
                    x0 + BORDER_SIZE,
                    y0 + BORDER_SIZE,
                    x0 + BORDER_SIZE + INNER_SIZE - 1,
                    y0 + BORDER_SIZE + INNER_SIZE - 1,
                ],
                fill=color,
            )

        # Draw Colorbar Sidebar
        cb_x0 = map_w + 50
        cb_x1 = cb_x0 + 60  # Bar is 60 pixels wide
        cb_y0 = 100
        cb_y1 = img_height - 100
        cb_height = cb_y1 - cb_y0

        for i in range(cb_height):
            ratio = 1.0 - (i / cb_height)
            color = interpolate_color(
                min_val + ratio * (max_val - min_val), min_val, max_val
            )
            draw.line([(cb_x0, cb_y0 + i), (cb_x1, cb_y0 + i)], fill=color, width=1)

        draw.rectangle([cb_x0, cb_y0, cb_x1, cb_y1], outline="white", width=4)

        num_ticks = 10
        if max_val > min_val:
            for i in range(num_ticks + 1):
                ratio = i / num_ticks
                val = min_val + ratio * (max_val - min_val)
                y_pos = cb_y1 - (ratio * cb_height)

                # Large clear tick marks
                draw.line([(cb_x1, y_pos), (cb_x1 + 30, y_pos)], fill="white", width=6)

                # Draw Text (Centered vertically using font offset logic)
                # Shifting up roughly half the font size centers the text along the tick line
                text_str = format_value(val)
                draw.text(
                    (cb_x1 + 45, y_pos - (font_size * 0.5)),
                    text_str,
                    fill="white",
                    font=font,
                )

        # Assemble robust filename
        fname = f"Heatmap_{area_label}_{dev_type}_Bit{int(bitval)}"
        if has_nvled_v:
            fname += f"_NVLED{nvled_v:.3f}V"
        fname += ".png"

        img.save(os.path.join(output_dir, fname))

    print(f"Done. Images saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate heatmaps from measurement CSV."
    )
    parser.add_argument("csv_file", help="Path to the measurement CSV file")
    args = parser.parse_args()
    generate_images(args.csv_file)
