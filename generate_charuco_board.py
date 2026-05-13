#!/usr/bin/env python3
"""
Generate and save a ChArUco board image for camera calibration.

Usage:
    python3 generate_charuco_board.py
    python3 generate_charuco_board.py --cols 7 --rows 5 --square 0.04 --marker 0.03 --dpi 300 --out board.png
"""

import argparse
import sys
import cv2
import numpy as np


# ── Board geometry defaults ────────────────────────────────────────────────────

# Marker size must be strictly less than square size so the white border
# around each ArUco marker is visible — detectors need that border to find corners.
DEFAULT_COLS        = 7        # chessboard columns (number of squares)
DEFAULT_ROWS        = 5        # chessboard rows
DEFAULT_SQUARE_M    = 0.04     # physical square side length in metres
DEFAULT_MARKER_M    = 0.03     # physical ArUco marker side length in metres
DEFAULT_DICT        = cv2.aruco.DICT_5X5_50
DEFAULT_DPI         = 300
DEFAULT_OUTPUT      = "charuco_board.png"


def parse_args():
    p = argparse.ArgumentParser(description="Generate a ChArUco calibration board.")
    p.add_argument("--cols",   type=int,   default=DEFAULT_COLS,     help="Number of squares (columns)")
    p.add_argument("--rows",   type=int,   default=DEFAULT_ROWS,     help="Number of squares (rows)")
    p.add_argument("--square", type=float, default=DEFAULT_SQUARE_M, help="Square side length in metres")
    p.add_argument("--marker", type=float, default=DEFAULT_MARKER_M, help="Marker side length in metres")
    p.add_argument("--dpi",    type=int,   default=DEFAULT_DPI,      help="Output resolution in DPI")
    p.add_argument("--out",    type=str,   default=DEFAULT_OUTPUT,   help="Output filename")
    return p.parse_args()


def validate(args):
    if args.marker >= args.square:
        sys.exit(f"Error: marker size ({args.marker} m) must be smaller than square size ({args.square} m).")

    # Each interior black square gets one ArUco marker.  The board has
    # floor(cols*rows / 2) such squares, so the dictionary must be large enough.
    num_markers = (args.cols * args.rows) // 2
    dict_obj    = cv2.aruco.getPredefinedDictionary(DEFAULT_DICT)
    if num_markers > dict_obj.bytesList.shape[0]:
        sys.exit(
            f"Error: board needs {num_markers} markers but DICT_5X5_50 only has "
            f"{dict_obj.bytesList.shape[0]}. Use a larger dictionary."
        )
    return dict_obj


def build_board(args, dictionary):
    return cv2.aruco.CharucoBoard(
        (args.cols, args.rows),
        args.square,
        args.marker,
        dictionary,
    )


def render(board, dpi, square_m):
    # Convert physical board size to pixels at the requested DPI.
    cols, rows    = board.getChessboardSize()
    board_w_m     = cols * square_m
    board_h_m     = rows * square_m
    metres_per_in = 0.0254
    px_w = int(round(board_w_m / metres_per_in * dpi))
    px_h = int(round(board_h_m / metres_per_in * dpi))

    return board.generateImage((px_w, px_h), marginSize=0, borderBits=1)


def add_info_strip(image, args):
    """Append a white strip below the board with key parameters printed on it.

    Printed on the physical page so there is no ambiguity when the board image
    is stored without a separate README.
    """
    strip_h = max(60, image.shape[0] // 15)
    strip   = np.full((strip_h, image.shape[1]), 255, dtype=np.uint8)

    text = (
        f"ChArUco {args.cols}x{args.rows}  |  "
        f"square={args.square*100:.1f} cm  |  "
        f"marker={args.marker*100:.1f} cm  |  "
        f"DICT_5X5_50  |  "
        f"{args.dpi} DPI"
    )
    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = strip_h / 90.0
    thickness  = max(1, strip_h // 40)
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Centre the text horizontally; align to bottom with a small margin.
    tx = (strip.shape[1] - tw) // 2
    ty = strip_h - (strip_h - th) // 3
    cv2.putText(strip, text, (tx, ty), font, font_scale, 0, thickness, cv2.LINE_AA)

    return np.vstack([image, strip])


def main():
    args       = parse_args()
    dictionary = validate(args)
    board      = build_board(args, dictionary)
    image      = render(board, args.dpi, args.square)
    image      = add_info_strip(image, args)

    cv2.imwrite(args.out, image)

    cols, rows = board.getChessboardSize()
    print(f"Saved: {args.out}")
    print(f"  Board   : {cols} x {rows} squares")
    print(f"  Square  : {args.square*100:.1f} cm")
    print(f"  Marker  : {args.marker*100:.1f} cm")
    print(f"  Size    : {image.shape[1]} x {image.shape[0]} px  @ {args.dpi} DPI")
    print(f"  Dict    : DICT_5X5_50")


if __name__ == "__main__":
    main()
