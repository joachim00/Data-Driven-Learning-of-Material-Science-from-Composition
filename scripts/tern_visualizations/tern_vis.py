#!/usr/bin/env python3
"""CLI utility for visualising ternary composition slices.

This script adapts the iterative analysis workflow from the exploratory
notebook into a reproducible command-line tool.  It loads composition,
absorption, and (optionally) prediction arrays, identifies the most
frequent ternary element combinations, filters the data, and produces a
grid of ternary plots coloured by model error metrics.

Examples:

	python scripts/tern_visualizations/tern_vis.py \
		--compositions data/compositions.npy \
		--absorption data/absorption.npy \
		--predictions data/predictions.npy \
		--element-names data/all_elements.txt \
		--top-n 12 --save ternaries.png

	python scripts/tern_visualizations/tern_vis.py \
		--compositions generated_samples/merged.h5::compositions \
		--absorption generated_samples/merged.h5::targets \
		--predictions generated_samples/merged.h5::predictions \
		--element-names data/all_elements.txt

The script accepts both .npy and .npz inputs (pass --*-key for the latter),
produces human-readable console summaries, and can save high-resolution
figures suitable for reports.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import h5py
import colorsys

try:  # Ensure mpltern is available and registers the ternary projection.
	import mpltern  # noqa: F401
except ImportError as exc:  # pragma: no cover - dependency check only
	raise SystemExit(
		"mpltern is required for ternary visualisation. Install it with `pip install mpltern`."
	) from exc


def _load_h5_chunk(args):
	"""Worker function to load a chunk of HDF5 images in parallel - OPTIMIZED.
	
	Must be at module level for multiprocessing.Pool to pickle it.
	"""
	h5_path, dataset_name, indices = args
	import h5py
	import numpy as np
	
	with h5py.File(h5_path, 'r', rdcc_nbytes=1024**3, rdcc_nslots=10000) as f:  # 1GB cache
		dataset = f[dataset_name]
		
		# Pre-allocate result array for speed
		sample_shape = dataset.shape[1:]
		result = np.empty((len(indices),) + sample_shape, dtype=dataset.dtype)
		
		# Check if indices are contiguous
		if len(indices) > 1 and np.all(np.diff(indices) == 1):
			# Contiguous read - MUCH faster
			result[:] = dataset[int(indices[0]):int(indices[-1])+1]
			return result
		
		# Try fancy indexing first
		try:
			result[:] = dataset[indices.tolist()]
			return result
		except (TypeError, KeyError, OSError):
			# Fallback: load one by one (slower but reliable)
			for i, idx in enumerate(indices):
				result[i] = dataset[int(idx)]
			return result


@dataclass
class TernarySlice:
	"""Container for a single ternary composition subset and its metadata."""

	combination: Tuple[int, ...]
	reduced_compositions: np.ndarray
	element_labels: Tuple[str, str, str]
	selection_indices: np.ndarray
	colors: Optional[np.ndarray] = None
	images: Optional[np.ndarray] = None  # Store cropped images for direct plotting


def split_path_and_key(spec: str, key: Optional[str]) -> Tuple[Path, Optional[str]]:
	"""Allow either a plain path or a `path::key` spec combined with CLI key."""

	if "::" in spec:
		path_str, inferred_key = spec.split("::", 1)
		return Path(path_str), inferred_key
	return Path(spec), key


def load_array(path: Path, key: Optional[str] = None, *, parse_json_compositions: bool = False, element_names: Optional[List[str]] = None) -> np.ndarray:
	"""Load a numpy array from .npy or .npz, optionally selecting a key.
	
	If parse_json_compositions is True, the array is expected to contain JSON strings
	representing compositions (e.g., '{"Fe": 0.5, "Ni": 0.5}'), which will be converted
	to numeric arrays using the provided element_names list.
	"""

	if not path.exists():
		raise FileNotFoundError(f"Input array not found: {path}")

	suffix = path.suffix.lower()

	if suffix == ".npy":
		array = np.load(path, allow_pickle=True)
		result = np.asarray(array)
	elif suffix == ".npz":
		data = np.load(path, allow_pickle=True)
		try:
			if key is None:
				available = ", ".join(sorted(data.files))
				raise ValueError(
					f"Array key required for npz file: {path}. Available keys: {available}"
				)
			result = np.asarray(data[key])
		finally:
			data.close()
	elif suffix in {".h5", ".hdf5", ".hdf"}:
		if key is None:
			raise ValueError(
				f"Dataset path required for HDF5 file: {path}. Use the syntax 'file.h5::dataset/name'."
			)
		with h5py.File(path, "r") as h5file:
			try:
				dataset = h5file[key]
			except KeyError as exc:
				available = list(h5file.keys())
				raise KeyError(
					f"Dataset '{key}' not found in {path}. Top-level datasets: {available}"
				) from exc
			result = np.asarray(dataset[()])
	else:
		raise ValueError(
			f"Unsupported file extension '{suffix}' for {path}. Expected .npy, .npz, or .h5/.hdf5/.hdf"
		)

	# Convert JSON compositions to numeric arrays if requested
	if parse_json_compositions:
		if element_names is None:
			raise ValueError("element_names must be provided when parse_json_compositions=True")
		result = _convert_json_compositions_to_numeric(result, element_names)

	return result


def _convert_json_compositions_to_numeric(json_array: np.ndarray, element_names: List[str], batch_size: int = 10000) -> np.ndarray:
	"""Convert an array of JSON composition strings to numeric arrays.
	
	Processes in batches to reduce memory footprint for large datasets.
	"""
	element_to_idx = {name: idx for idx, name in enumerate(element_names)}
	num_elements = len(element_names)
	total = len(json_array)
	
	# Pre-allocate output array
	numeric_compositions = np.zeros((total, num_elements), dtype=np.float32)
	
	for start_idx in range(0, total, batch_size):
		end_idx = min(start_idx + batch_size, total)
		if start_idx % 100000 == 0 and start_idx > 0:
			print(f"  Converting compositions: {start_idx}/{total} ({100*start_idx//total}%)")
		
		for i in range(start_idx, end_idx):
			raw = json_array[i]
			# Handle bytes or string
			s = raw.decode('utf-8') if isinstance(raw, bytes) else str(raw)
			try:
				comp_dict = json.loads(s)
			except json.JSONDecodeError:
				# Fallback for single-quote JSON
				try:
					comp_dict = json.loads(s.replace("'", '"'))
				except Exception as e:
					if i < 5:  # Only warn for first few errors
						print(f"Warning: failed to parse composition at index {i}: {e}")
					continue
			
			# Convert to numeric array
			for element, fraction in comp_dict.items():
				if element in element_to_idx:
					numeric_compositions[i, element_to_idx[element]] = float(fraction)
	
	return numeric_compositions


def load_element_names(path: Optional[Path], expected: Optional[int]) -> List[str]:
	"""Load element labels from JSON or plain-text file; fallback to indices.
	
	If expected is None, load all available names without validation.
	"""

	if path is None:
		if expected is None:
			raise ValueError("Cannot generate default element names without expected count.")
		return [f"E{i}" for i in range(expected)]

	if not path.exists():
		raise FileNotFoundError(f"Element names file not found: {path}")

	if path.suffix.lower() in {".json", ".json5"}:
		names = json.loads(path.read_text(encoding="utf-8"))
	else:
		names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

	if expected is not None and len(names) != expected:
		raise ValueError(
			f"Expected {expected} element names, received {len(names)} from {path}"
		)
	return names


def load_reference_compositions(path: Optional[Path]) -> Optional[np.ndarray]:
	"""Load reference compositions (list of lists of floats) if provided."""

	if path is None:
		return None
	if not path.exists():
		raise FileNotFoundError(f"Reference compositions file not found: {path}")

	data = json.loads(path.read_text(encoding="utf-8"))
	arr = np.asarray(data)
	if arr.ndim != 2:
		raise ValueError("Reference compositions must be a 2D array-like structure.")
	return arr


COLOR_MODE_MEAN_RGB = "mean_rgb"
COLOR_MODE_DOMINANT_HUE = "dominant_hue"
COLOR_MODE_GRAYSCALE = "grayscale"
COLOR_MODES_ALL = [COLOR_MODE_MEAN_RGB, COLOR_MODE_DOMINANT_HUE, COLOR_MODE_GRAYSCALE]


def _prepare_image_array(image: np.ndarray) -> np.ndarray:
	"""Convert array to float32 (H, W, C) in [0, 1]."""

	arr = np.asarray(image)
	if arr.ndim == 3:
		# Handle channel-first layouts by detecting small channel dimension
		if arr.shape[0] in {1, 3} and arr.shape[0] < arr.shape[-1]:
			arr = np.transpose(arr, (1, 2, 0))
	elif arr.ndim == 2:
		arr = arr[..., None]
	else:
		raise ValueError(f"Unsupported image shape {arr.shape}. Expected 2D or 3D array.")

	arr = arr.astype(np.float32, copy=False)
	if np.issubdtype(image.dtype, np.integer) or (arr.size > 0 and np.nanmax(arr) > 1.5):
		arr /= 255.0
	return arr


def _center_crop(arr: np.ndarray, crop_size: int) -> np.ndarray:
	if crop_size <= 0:
		return arr
	h, w = arr.shape[:2]
	crop = min(crop_size, h, w)
	h0 = max(0, (h - crop) // 2)
	w0 = max(0, (w - crop) // 2)
	return arr[h0 : h0 + crop, w0 : w0 + crop, ...]


def compute_mean_rgb(image: np.ndarray, crop_size: int) -> np.ndarray:
	arr = _center_crop(_prepare_image_array(image), crop_size)
	if arr.size == 0:
		return np.array([0.5, 0.5, 0.5], dtype=np.float32)
	if arr.shape[2] < 3:
		arr = np.repeat(arr, 3, axis=2)
	rgb = arr.reshape(-1, arr.shape[2]).mean(axis=0)
	return rgb.astype(np.float32)


def compute_mean_rgb_batch(images: np.ndarray, crop_size: int) -> np.ndarray:
	"""Vectorized batch computation of mean RGB colors."""
	batch_size = len(images)
	colors = np.zeros((batch_size, 3), dtype=np.float32)
	
	for i, image in enumerate(images):
		colors[i] = compute_mean_rgb(image, crop_size)
	
	return colors


def compute_grayscale(image: np.ndarray, crop_size: int) -> np.ndarray:
	arr = _center_crop(_prepare_image_array(image), crop_size)
	if arr.size == 0:
		return np.array([0.5, 0.5, 0.5], dtype=np.float32)
	value = float(arr.mean())
	return np.array([value, value, value], dtype=np.float32)


def compute_grayscale_batch(images: np.ndarray, crop_size: int) -> np.ndarray:
	"""Vectorized batch computation of grayscale colors."""
	batch_size = len(images)
	colors = np.zeros((batch_size, 3), dtype=np.float32)
	
	for i, image in enumerate(images):
		colors[i] = compute_grayscale(image, crop_size)
	
	return colors


def _rgb_to_hsv(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	maxc = arr.max(axis=2)
	minc = arr.min(axis=2)
	v = maxc
	deltac = maxc - minc
	s = np.zeros_like(maxc, dtype=np.float32)
	mask = maxc > 0
	s[mask] = deltac[mask] / maxc[mask]

	h = np.zeros_like(maxc, dtype=np.float32)
	nonzero_mask = deltac > 0
	rc = np.zeros_like(maxc, dtype=np.float32)
	gc = np.zeros_like(maxc, dtype=np.float32)
	bc = np.zeros_like(maxc, dtype=np.float32)
	rc[nonzero_mask] = (maxc[nonzero_mask] - arr[..., 0][nonzero_mask]) / deltac[nonzero_mask]
	gc[nonzero_mask] = (maxc[nonzero_mask] - arr[..., 1][nonzero_mask]) / deltac[nonzero_mask]
	bc[nonzero_mask] = (maxc[nonzero_mask] - arr[..., 2][nonzero_mask]) / deltac[nonzero_mask]

	rmask = (arr[..., 0] == maxc) & nonzero_mask
	gmask = (arr[..., 1] == maxc) & nonzero_mask
	bmask = (arr[..., 2] == maxc) & nonzero_mask

	h[rmask] = (bc[rmask] - gc[rmask])
	h[gmask] = 2.0 + (rc[gmask] - bc[gmask])
	h[bmask] = 4.0 + (gc[bmask] - rc[bmask])

	h = (h / 6.0) % 1.0
	return h, s, v


def compute_dominant_hue(image: np.ndarray, crop_size: int, hue_bins: int) -> np.ndarray:
	arr = _center_crop(_prepare_image_array(image), crop_size)
	if arr.size == 0:
		return np.array([0.5, 0.5, 0.5], dtype=np.float32)
	if arr.shape[2] < 3:
		arr = np.repeat(arr, 3, axis=2)

	h, s, v = _rgb_to_hsv(arr)
	mask = (s > 0.15) & (v > 0.1)
	if not np.any(mask):
		mask = np.ones_like(h, dtype=bool)

	hist, edges = np.histogram(h[mask], bins=hue_bins, range=(0.0, 1.0))
	if np.all(hist == 0):
		return arr.reshape(-1, arr.shape[2]).mean(axis=0).astype(np.float32)
	dominant_bin = int(np.argmax(hist))
	hue_value = (edges[dominant_bin] + edges[dominant_bin + 1]) * 0.5
	rgb = colorsys.hsv_to_rgb(hue_value, 1.0, 1.0)
	return np.array(rgb, dtype=np.float32)


def compute_dominant_hue_batch(images: np.ndarray, crop_size: int, hue_bins: int) -> np.ndarray:
	"""Vectorized batch computation of dominant hue colors."""
	batch_size = len(images)
	colors = np.zeros((batch_size, 3), dtype=np.float32)
	
	for i, image in enumerate(images):
		colors[i] = compute_dominant_hue(image, crop_size, hue_bins)
	
	return colors


class ImageColorProvider:
	"""Lazily compute per-sample colours from an image dataset."""

	def __init__(
		self,
		path: Path,
		key: Optional[str],
		color_mode: str,
		*,
		crop_size: int,
		hue_bins: int,
	) -> None:
		self.path = path
		self.key = key
		self.color_mode = color_mode
		self.crop_size = crop_size
		self.hue_bins = hue_bins

		self._h5: Optional[h5py.File] = None
		self._dataset = None
		self._np_array = None
		self._npz_file = None

	def __enter__(self) -> "ImageColorProvider":
		self.open()
		return self

	def __exit__(self, exc_type, exc, tb) -> None:
		self.close()

	def open(self) -> None:
		suffix = self.path.suffix.lower()
		if suffix == ".npy":
			self._np_array = np.load(self.path, mmap_mode="r")
		elif suffix == ".npz":
			self._npz_file = np.load(self.path, allow_pickle=False)
			key = self.key
			if key is None:
				available = ", ".join(sorted(self._npz_file.files))
				raise ValueError(
					f"Dataset/key required for npz file {self.path}. Available keys: {available}"
				)
			self._np_array = self._npz_file[key]
		elif suffix in {".h5", ".hdf5", ".hdf"}:
			key = self.key
			if key is None:
				raise ValueError(
					f"Dataset path required for HDF5 file {self.path}. Use 'file.h5::dataset/path'."
				)
			self._h5 = h5py.File(self.path, "r")
			try:
				self._dataset = self._h5[key]
			except KeyError as exc:
				available = list(self._h5.keys())
				self._h5.close()
				self._h5 = None
				raise KeyError(
					f"Dataset '{key}' not found in {self.path}. Top-level datasets: {available}"
				) from exc
		else:
			raise ValueError(
				f"Unsupported image source '{self.path}'. Expected .npy, .npz, or .h5/.hdf5/.hdf"
			)

	def close(self) -> None:
		if self._h5 is not None:
			self._h5.close()
			self._h5 = None
		if self._npz_file is not None:
			self._npz_file.close()
			self._npz_file = None

	def _get_item(self, index: int) -> np.ndarray:
		if self._dataset is not None:
			return self._dataset[index]
		if self._np_array is not None:
			return self._np_array[index]
		raise RuntimeError("ImageColorProvider is not initialised. Call open() first.")

	def _compute_color(self, image: np.ndarray) -> np.ndarray:
		mode = self.color_mode
		if mode == "mean_rgb":
			return compute_mean_rgb(image, self.crop_size)
		if mode == "grayscale":
			return compute_grayscale(image, self.crop_size)
		if mode == "dominant_hue":
			return compute_dominant_hue(image, self.crop_size, self.hue_bins)
		raise ValueError(f"Unsupported color mode '{mode}' for image colours.")
	
	def _compute_color_batch(self, images: np.ndarray) -> np.ndarray:
		"""Batch compute colors for multiple images."""
		mode = self.color_mode
		if mode == "mean_rgb":
			return compute_mean_rgb_batch(images, self.crop_size)
		if mode == "grayscale":
			return compute_grayscale_batch(images, self.crop_size)
		if mode == "dominant_hue":
			return compute_dominant_hue_batch(images, self.crop_size, self.hue_bins)
		raise ValueError(f"Unsupported color mode '{mode}' for image colours.")

	def fetch(self, indices: np.ndarray, batch_size: int = 300) -> np.ndarray:
		"""Fetch colors for given indices in batches to reduce memory usage."""
		import time
		
		if indices.size == 0:
			return np.empty((0, 3), dtype=np.float32)
		
		total = len(indices)
		colors = np.zeros((total, 3), dtype=np.float32)
		
		print(f"  Computing colors for {total} samples in batches of {batch_size}...")
		start_time = time.time()
		
		for start_idx in range(0, total, batch_size):
			end_idx = min(start_idx + batch_size, total)
			batch_start = time.time()
			
			if start_idx % 100 == 0 and start_idx > 0:
				elapsed = time.time() - start_time
				print(f"  Computing colors: {start_idx}/{total} ({100*start_idx//total}%) - {elapsed:.2f}s elapsed")
			
			batch_indices = indices[start_idx:end_idx]
			
			# Load images one by one (HDF5 fancy indexing can be very slow for scattered indices)
			load_start = time.time()
			batch_images = []
			for idx in batch_indices:
				if self._dataset is not None:
					batch_images.append(self._dataset[int(idx)])
				elif self._np_array is not None:
					batch_images.append(self._np_array[int(idx)])
				else:
					raise RuntimeError("ImageColorProvider is not initialised.")
			load_time = time.time() - load_start
			
			# Compute colors for the entire batch at once
			compute_start = time.time()
			batch_images = np.array(batch_images)
			batch_colors = self._compute_color_batch(batch_images)
			colors[start_idx:end_idx] = batch_colors
			compute_time = time.time() - compute_start
			
			if start_idx == 0:  # Print timing breakdown for first batch
				print(f"    First batch timing: load={load_time:.3f}s, compute={compute_time:.3f}s")
		
		total_time = time.time() - start_time
		print(f"  Completed color computation in {total_time:.2f}s ({total_time/total:.3f}s per sample)")
		
		return colors
	
	def fetch_images(self, indices: np.ndarray, batch_size: int = 300, parallel: bool = True, num_workers: int = 4) -> np.ndarray:
		"""Fetch raw images for given indices in batches.
		
		Args:
			indices: Array of indices to fetch
			batch_size: Number of images to load per batch
			parallel: If True, use parallel loading (faster for scattered indices)
			num_workers: Number of parallel workers to use (default: 4)
		"""
		import time
		
		if indices.size == 0:
			return np.empty((0, 0, 0), dtype=np.float32)
		
		total = len(indices)
		
		# OPTIMIZATION 1: Sort indices for better HDF5 cache locality
		sort_start = time.time()
		sort_order = np.argsort(indices)
		sorted_indices = indices[sort_order]
		print(f"  Sorted {total} indices in {time.time() - sort_start:.3f}s")
		
		print(f"  Loading {total} images...")
		start_time = time.time()
		
		# OPTIMIZATION 2: Try parallel loading if requested
		if parallel and total > 50:  # Increased threshold for parallel overhead
			try:
				images_sorted = self._fetch_images_parallel(sorted_indices, batch_size, num_workers)
			except Exception as e:
				print(f"    Parallel loading failed ({e}), falling back to sequential...")
				images_sorted = self._fetch_images_sequential(sorted_indices, batch_size)
		else:
			images_sorted = self._fetch_images_sequential(sorted_indices, batch_size)
		
		# OPTIMIZATION 3: Faster unsort using advanced indexing
		unsort_start = time.time()
		unsort_order = np.empty_like(sort_order)
		unsort_order[sort_order] = np.arange(len(sort_order))
		images_list = images_sorted[unsort_order]
		print(f"  Restored original order in {time.time() - unsort_start:.3f}s")
		
		total_time = time.time() - start_time
		print(f"  Completed image loading in {total_time:.2f}s ({total_time/total:.4f}s per image, {total/total_time:.1f} img/s)")
		
		return images_list
	
	def _fetch_images_sequential(self, indices: np.ndarray, batch_size: int) -> np.ndarray:
		"""Sequential image loading (fallback method) with optimizations."""
		import time
		total = len(indices)
		
		# Pre-allocate result array - MAJOR SPEEDUP
		if self._dataset is not None:
			sample_shape = self._dataset.shape[1:]
			sample_dtype = self._dataset.dtype
		elif self._np_array is not None:
			sample_shape = self._np_array.shape[1:]
			sample_dtype = self._np_array.dtype
		else:
			raise RuntimeError("ImageColorProvider is not initialised.")
		
		result = np.empty((total,) + sample_shape, dtype=sample_dtype)
		
		# Try to load entire batch at once if indices are contiguous
		if self._is_contiguous(indices):
			print(f"    Detected contiguous indices, loading in one operation...")
			if self._dataset is not None:
				result[:] = self._dataset[int(indices[0]):int(indices[-1])+1]
			else:
				result[:] = self._np_array[int(indices[0]):int(indices[-1])+1]
			return result
		
		# Load in batches
		for start_idx in range(0, total, batch_size):
			end_idx = min(start_idx + batch_size, total)
			batch_start_time = time.time()
			
			batch_indices = indices[start_idx:end_idx]
			
			# Try fancy indexing first (single HDF5 call)
			try:
				if self._dataset is not None:
					# Convert to list for h5py fancy indexing
					result[start_idx:end_idx] = self._dataset[batch_indices.tolist()]
				elif self._np_array is not None:
					result[start_idx:end_idx] = self._np_array[batch_indices]
			except (TypeError, KeyError, OSError) as e:
				# Fallback: load one by one (HDF5 fancy indexing can fail)
				if start_idx == 0:  # Only print once
					print(f"    Fancy indexing failed ({type(e).__name__}), loading individually...")
				for i, idx in enumerate(batch_indices):
					if self._dataset is not None:
						result[start_idx + i] = self._dataset[int(idx)]
					elif self._np_array is not None:
						result[start_idx + i] = self._np_array[int(idx)]
			
			batch_time = time.time() - batch_start_time
			rate = len(batch_indices) / batch_time
			if (start_idx // batch_size) % 5 == 0 or end_idx == total:  # Print every 5th batch
				print(f"    Batch {start_idx}-{end_idx}: {batch_time:.2f}s ({rate:.1f} img/s)")
		
		return result
	
	def _is_contiguous(self, indices: np.ndarray) -> bool:
		"""Check if indices are contiguous."""
		if len(indices) <= 1:
			return True
		return np.all(np.diff(indices) == 1)
	
	def _fetch_images_parallel(self, indices: np.ndarray, batch_size: int, num_workers: int = 4) -> np.ndarray:
		"""Parallel image loading using multiprocessing - OPTIMIZED."""
		from multiprocessing import Pool, cpu_count
		import time
		
		if self._np_array is not None:
			# Numpy arrays can be shared directly
			return self._np_array[indices]
		
		if self._dataset is None:
			raise RuntimeError("ImageColorProvider is not initialised.")
		
		# For HDF5, we need to pass the file path and reopen in each worker
		h5_path = self._dataset.file.filename
		dataset_name = self._dataset.name
		
		# Use user-specified num_workers, limited by CPU count
		n_workers = min(num_workers, cpu_count())
		print(f"    Using parallel loading with {n_workers} workers...")
		
		# OPTIMIZATION: Use larger chunks to reduce overhead
		# Aim for at least 50 images per chunk, but not more than 2x num_workers chunks
		min_chunk_size = 50
		max_chunks = n_workers * 2
		chunk_size = max(min_chunk_size, len(indices) // max_chunks)
		
		chunks = [indices[i:i+chunk_size] for i in range(0, len(indices), chunk_size)]
		print(f"    Split into {len(chunks)} chunks of ~{chunk_size} images each")
		
		# Create worker arguments
		worker_args = [(h5_path, dataset_name, chunk) for chunk in chunks]
		
		# Parallel load
		start_time = time.time()
		with Pool(processes=n_workers) as pool:
			results = pool.map(_load_h5_chunk, worker_args)
		
		parallel_time = time.time() - start_time
		rate = len(indices) / parallel_time
		print(f"    Parallel loading completed in {parallel_time:.2f}s ({rate:.1f} img/s)")
		
		# Concatenate results - use vstack for better performance with large arrays
		return np.vstack(results) if len(results) > 1 else results[0]

def identify_most_popular_combination(
	datapoints: np.ndarray, num_elements: int, top_n: int, sample_size: Optional[int] = None, min_samples: int = 1
) -> List[Tuple[int, ...]]:
	"""Return the most frequent element index combinations.
	
	If sample_size is provided and dataset is large, sample randomly to reduce processing time.
	
	Args:
		datapoints: 2D array of compositions (samples x elements)
		num_elements: Number of elements per combination (e.g., 3 for ternary)
		top_n: Number of top combinations to return
		sample_size: Optional sample size for large datasets (default: None = use all)
		min_samples: Minimum number of samples required for a combination to be included (default: 1)
	"""
	import time

	if datapoints.ndim != 2:
		raise ValueError("datapoints must be a 2D array (samples x elements)")

	start_time = time.time()
	
	# OPTIMIZATION 1: Pre-filter to only compositions with exactly num_elements
	# This is much faster than checking in the loop
	element_counts = np.count_nonzero(datapoints, axis=1)
	valid_mask = element_counts == num_elements
	valid_datapoints = datapoints[valid_mask]
	
	print(f"  Found {len(valid_datapoints):,} compositions with exactly {num_elements} elements (from {len(datapoints):,} total)")
	
	if len(valid_datapoints) == 0:
		raise ValueError(
			f"No compositions contained exactly {num_elements} non-zero elements."
		)
	
	# For large datasets, sample to speed up combination finding
	if sample_size and len(valid_datapoints) > sample_size:
		print(f"  Sampling {sample_size:,} compositions from {len(valid_datapoints):,} valid to find popular combinations...")
		indices = np.random.choice(len(valid_datapoints), size=sample_size, replace=False)
		sample_data = valid_datapoints[indices]
	else:
		sample_data = valid_datapoints

	# OPTIMIZATION 2: Vectorized combination extraction
	print(f"  Extracting combinations from {len(sample_data):,} compositions...")
	valid_combinations = [
		tuple(sorted(np.nonzero(composition)[0]))
		for composition in sample_data
	]

	from collections import Counter

	combination_counts = Counter(valid_combinations)
	
	# OPTIMIZATION 3: Filter by minimum sample count
	if min_samples > 1:
		filtered_counts = {combo: count for combo, count in combination_counts.items() if count >= min_samples}
		print(f"  Filtered from {len(combination_counts)} to {len(filtered_counts)} combinations with >={min_samples} samples")
		combination_counts = Counter(filtered_counts)
	
	if not combination_counts:
		raise ValueError(f"No combinations found with at least {min_samples} samples.")
	
	most_common = [combo for combo, _ in combination_counts.most_common(top_n)]
	most_common_counts = [count for _, count in combination_counts.most_common(top_n)]

	elapsed = time.time() - start_time
	print(f"  Top {len(most_common)} combinations found in {elapsed:.2f}s:")
	for i, (combo, count) in enumerate(zip(most_common, most_common_counts), 1):
		if i <= 10 or i > len(most_common) - 3:  # Show first 10 and last 3
			print(f"    {i}. {combo}: {count} samples")
		elif i == 11:
			print(f"    ... ({len(most_common) - 13} more) ...")
	
	return most_common


def filter_compositions_systems(
	datapoints: np.ndarray,
	combinations_list: Iterable[Sequence[int]],
	min_elements_present: int,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Filter compositions matching any combination and return indices.
	
	OPTIMIZED: Uses vectorized operations for much faster filtering on large datasets.
	"""
	import time
	
	start_time = time.time()
	filtered_data: List[np.ndarray] = []
	selected_indices: List[int] = []

	combinations_set_list = [set(combo) for combo in combinations_list]

	# OPTIMIZATION: For single combination (common case), use vectorized approach
	if len(combinations_set_list) == 1:
		combination_set = combinations_set_list[0]
		combination_indices = np.array(list(combination_set))
		
		# Check if compositions have non-zero values at the combination indices
		has_elements = datapoints[:, combination_indices] > 0
		
		# Count how many combination elements are present in each composition
		num_present = has_elements.sum(axis=1)
		
		# Count total non-zero elements in each composition
		total_nonzero = (datapoints > 0).sum(axis=1)
		
		# Filter: has enough combination elements AND has no other elements
		valid_mask = (num_present >= min_elements_present) & (num_present == total_nonzero)
		
		selected_indices = np.where(valid_mask)[0]
		filtered_data = datapoints[valid_mask]
		
		elapsed = time.time() - start_time
		# print(f"  Filtered {len(datapoints):,} compositions to {len(selected_indices):,} matches in {elapsed:.3f}s")
		
		return filtered_data, selected_indices
	
	# Fallback for multiple combinations (less common, use original logic)
	for idx, composition in enumerate(datapoints):
		non_zero_indices = set(np.nonzero(composition)[0])
		for combination_set in combinations_set_list:
			common = non_zero_indices & combination_set
			if (
				len(common) >= min_elements_present
				and len(common) == len(non_zero_indices)
			):
				filtered_data.append(composition)
				selected_indices.append(idx)
				break

	if not filtered_data:
		return (
			np.empty((0, datapoints.shape[1]), dtype=datapoints.dtype),
			np.empty((0,), dtype=int),
		)

	elapsed = time.time() - start_time
	# print(f"  Filtered {len(datapoints):,} compositions to {len(selected_indices):,} matches in {elapsed:.3f}s")
	
	return (
		np.asarray(filtered_data),
		np.asarray(selected_indices, dtype=int),
	)


def reduce_compositions(
	compositions: np.ndarray,
	*,
	reference_compositions: Optional[np.ndarray] = None,
	element_names: Optional[Sequence[str]] = None,
	num_elements: int = 3,
	min_elements_present: int = 2,
	top_n: int = 1,
	max_samples_per_combination: Optional[int] = None,
	min_samples_per_combination: int = 1,
	sample_size: Optional[int] = None,
) -> List[TernarySlice]:
	"""Reduce the dataset to the most relevant ternary combinations.
	
	Args:
		compositions: 2D array of all compositions
		reference_compositions: Optional specific compositions to use instead of finding popular ones
		element_names: Optional list of element names for labeling
		num_elements: Number of elements per combination (default: 3 for ternary)
		min_elements_present: Minimum elements from combination required in a sample (default: 2)
		top_n: Number of top combinations to find (default: 1)
		max_samples_per_combination: Optional limit on samples per combination
		min_samples_per_combination: Minimum samples required for a combination to be included (default: 1)
		sample_size: Optional sample size for finding popular combinations (default: None = use all)
	"""
	import time

	if compositions.ndim != 2:
		raise ValueError("compositions must be 2D (samples x elements)")

	total_start = time.time()
	
	if reference_compositions is None:
		combinations = identify_most_popular_combination(
			compositions, 
			num_elements=num_elements, 
			top_n=top_n, 
			sample_size=sample_size or 50000,
			min_samples=min_samples_per_combination
		)
	else:
		combinations = [tuple(np.nonzero(row)[0]) for row in reference_compositions]

	slices: List[TernarySlice] = []
	
	print(f"\nFiltering compositions for {len(combinations)} combinations...")

	for combo_idx, combination in enumerate(combinations, 1):
		combo_start = time.time()
		
		filtered_comp, selection = filter_compositions_systems(
			compositions, [combination], min_elements_present
		)

		if filtered_comp.size == 0:
			print(f"  [{combo_idx}/{len(combinations)}] Combination {combination} yielded no samples after filtering; skipping.")
			continue

		# Limit samples if requested (for large datasets)
		if max_samples_per_combination and len(filtered_comp) > max_samples_per_combination:
			# print(f"  Limiting to {max_samples_per_combination} samples (from {len(filtered_comp)} total)")
			sample_indices = np.random.choice(len(filtered_comp), size=max_samples_per_combination, replace=False)
			sample_indices = np.sort(sample_indices)  # Keep order for consistency
			filtered_comp = filtered_comp[sample_indices]
			selection = selection[sample_indices]

		reduced = filtered_comp[:, list(combination)]
		labels = tuple(
			element_names[i] if element_names else str(i) for i in combination
		)

		combo_time = time.time() - combo_start
		
		# Show progress for every combination when there are many
		if len(combinations) > 20:
			if combo_idx % 50 == 0 or combo_idx <= 5 or combo_idx > len(combinations) - 3:
				print(f"  [{combo_idx}/{len(combinations)}] {combination} -> {len(reduced)} samples ({combo_time:.2f}s)")
		else:
			print(f"  [{combo_idx}/{len(combinations)}] Combination {combination} -> {len(reduced)} samples")

		slices.append(
			TernarySlice(
				combination=tuple(combination),
				reduced_compositions=reduced,
				element_labels=labels,  # type: ignore[arg-type]
				selection_indices=selection,
			)
		)

	if not slices:
		raise ValueError("No ternary combinations produced data; check filtering parameters.")

	total_time = time.time() - total_start
	print(f"\nCompleted filtering {len(slices)} combinations in {total_time:.1f}s")
	
	return slices


def trinary_plot_image_colours(
	reduced_data: np.ndarray,
	colours: np.ndarray,
	element_labels: Sequence[str],
	*,
	ax: Optional[plt.Axes] = None,
	gridsize: int = 10,
	alpha: float = 0.85,
	font_size: float = 10.0,
	line_width: float = 1.0,
) -> plt.Collection:
	"""Scatter a ternary plot using precomputed RGB colours."""

	if reduced_data.shape[1] != 3:
		raise ValueError("reduced_data must have 3 columns for ternary plotting")

	if colours.ndim != 2 or colours.shape[1] != 3:
		raise ValueError("colours must be an (N, 3) RGB array")

	if ax is None:
		ax = plt.subplot(projection="ternary")

	el1, el2, el3 = reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2]
	ax.set_tlabel(element_labels[0], fontsize=font_size)
	ax.set_llabel(element_labels[1], fontsize=font_size)
	ax.set_rlabel(element_labels[2], fontsize=font_size)
	
	# Set tick label font size and line widths
	ax.tick_params(axis='both', which='major', labelsize=font_size * 0.8, width=line_width)
	for spine in ax.spines.values():
		spine.set_linewidth(line_width)

	# Don't plot hexbin density background - removed to show only colored scatter points
	collection = ax.scatter(
		el1,
		el2,
		el3,
		s=150,
		c=colours,
		alpha=alpha,
		edgecolors="none",
	)
	return collection


def trinary_plot_with_images(
	reduced_data: np.ndarray,
	images: np.ndarray,
	element_labels: Sequence[str],
	*,
	ax: Optional[plt.Axes] = None,
	gridsize: int = 10,
	image_size: float = 0.03,
	crop_size: int = 36,
	font_size: float = 10.0,
	line_width: float = 1.0,
) -> None:
	"""Plot a ternary diagram with cropped images at each data point.
	
	Args:
		reduced_data: (N, 3) array of ternary compositions
		images: (N, H, W, C) array of images to crop and display
		element_labels: Labels for the three ternary axes
		ax: Matplotlib axes with ternary projection
		gridsize: Grid size for hexbin background
		image_size: Size of displayed images in axis coordinates (0.03 = ~3% of axis)
		crop_size: Size to crop from center of each image
		font_size: Font size for axis labels
		line_width: Line width for axes and ticks
	"""
	from matplotlib.offsetbox import OffsetImage, AnnotationBbox
	
	if reduced_data.shape[1] != 3:
		raise ValueError("reduced_data must have 3 columns for ternary plotting")

	if ax is None:
		ax = plt.subplot(projection="ternary")

	el1, el2, el3 = reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2]
	ax.set_tlabel(element_labels[0], fontsize=font_size)
	ax.set_llabel(element_labels[1], fontsize=font_size)
	ax.set_rlabel(element_labels[2], fontsize=font_size)
	
	# Set tick label font size and line widths
	ax.tick_params(axis='both', which='major', labelsize=font_size * 0.8, width=line_width)
	for spine in ax.spines.values():
		spine.set_linewidth(line_width)

	# Don't plot hexbin density background - removed to show only images
	
	# Debug: Print composition range
	print(f"  Composition ranges: el1=[{el1.min():.3f}, {el1.max():.3f}], el2=[{el2.min():.3f}, {el2.max():.3f}], el3=[{el3.min():.3f}, {el3.max():.3f}]")
	
	# Add cropped images at each point
	print(f"  Adding {len(reduced_data)} images to plot...")
	
	# Pre-compute ALL coordinate transformations at once (major speedup!)
	# Create a single scatter plot to get all transformed coordinates
	temp_scatter = ax.scatter(el1, el2, el3, s=0, alpha=0)
	xy_positions = temp_scatter.get_offsets()
	temp_scatter.remove()
	
	# Calculate zoom once
	fig = ax.get_figure()
	if fig:
		fig_width_inch, fig_height_inch = fig.get_size_inches()
		dpi = fig.dpi
		bbox = ax.get_position()
		subplot_width_inch = bbox.width * fig_width_inch
		subplot_height_inch = bbox.height * fig_height_inch
		subplot_size_inch = min(subplot_width_inch, subplot_height_inch)
		zoom = (image_size * subplot_size_inch * dpi) / crop_size
	else:
		zoom = image_size * 10  # Fallback
	
	for i in range(len(reduced_data)):
		if i % 50 == 0 and i > 0:
			print(f"    Added {i}/{len(reduced_data)} images...")
		
		# Crop center of image
		img = _prepare_image_array(images[i])
		cropped = _center_crop(img, crop_size)
		
		# Normalize image to [0, 1] if needed
		if cropped.max() > 1.0:
			cropped = cropped / 255.0
		
		imagebox = OffsetImage(cropped, zoom=zoom)
		
		# Use pre-computed coordinate
		if i < len(xy_positions):
			xy_pos = xy_positions[i]
		else:
			# Fallback
			xy_pos = (el2[i] + el3[i]/2, el3[i] * np.sqrt(3)/2)
		
		ab = AnnotationBbox(
			imagebox,
			xy_pos,
			xycoords='data',
			frameon=False,
			pad=0,
			box_alignment=(0.5, 0.5)
		)
		ax.add_artist(ab)
	
	print(f"  All {len(reduced_data)} images added to plot!")


def plot_single_ternary(
	slice_data: TernarySlice,
	ax: plt.Axes,
	*,
	gridsize: int = 10,
	color_alpha: float = 0.85,
	font_size: float = 10.0,
	line_width: float = 1.0,
) -> plt.Collection:
	"""Plot a single ternary slice with image-based colors."""

	if slice_data.colors is None:
		raise ValueError("No colours were computed for this slice.")
	
	return trinary_plot_image_colours(
		slice_data.reduced_compositions,
		slice_data.colors,
		slice_data.element_labels,
		ax=ax,
		gridsize=gridsize,
		alpha=color_alpha,
		font_size=font_size,
		line_width=line_width,
	)


def plot_single_ternary_with_images(
	slice_data: TernarySlice,
	ax: plt.Axes,
	*,
	gridsize: int = 10,
	image_size: float = 0.03,
	crop_size: int = 36,
	font_size: float = 10.0,
	line_width: float = 1.0,
) -> None:
	"""Plot a single ternary slice with actual cropped images at each point."""

	if slice_data.images is None:
		raise ValueError("No images were stored for this slice.")
	
	trinary_plot_with_images(
		slice_data.reduced_compositions,
		slice_data.images,
		slice_data.element_labels,
		ax=ax,
		gridsize=gridsize,
		image_size=image_size,
		crop_size=crop_size,
		font_size=font_size,
		line_width=line_width,
	)


def plot_all_ternary(
	slices: List[TernarySlice],
	*,
	fig_size: Tuple[float, float] = (30.0, 30.0),
	nrow: Optional[int] = None,
	ncol: Optional[int] = None,
	max_plots: Optional[int] = None,
	save_path: Optional[Path] = None,
	dpi: int = 200,
	gridsize: int = 10,
	color_alpha: float = 0.85,
	font_size: float = 10.0,
	line_width: float = 1.0,
) -> List[int]:
	"""Render a grid of ternary plots and optionally save to disk."""

	if not slices:
		raise ValueError("No ternary slices to plot.")

	order = list(range(len(slices)))

	if max_plots is not None:
		order = order[:max_plots]

	total_plots = len(order)
	if total_plots == 0:
		raise ValueError("No plots selected after applying max_plots filter.")

	if nrow is None or ncol is None:
		ncol = ncol or math.ceil(math.sqrt(total_plots))
		nrow = nrow or math.ceil(total_plots / ncol)

	fig = plt.figure(figsize=fig_size)

	for pos, idx in enumerate(order, start=1):
		ax = fig.add_subplot(nrow, ncol, pos, projection="ternary")
		plot_single_ternary(
			slices[idx],
			ax,
			gridsize=gridsize,
			color_alpha=color_alpha,
			font_size=font_size,
			line_width=line_width,
		)

	fig.tight_layout()
	fig.subplots_adjust(wspace=0.5, hspace=0.5)

	if save_path:
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
		print(f"Saved ternary grid to {save_path}")
	else:
		plt.show()

	plt.close(fig)
	return order


def plot_all_ternary_with_images(
	slices: List[TernarySlice],
	*,
	fig_size: Tuple[float, float] = (30.0, 30.0),
	nrow: Optional[int] = None,
	ncol: Optional[int] = None,
	max_plots: Optional[int] = None,
	save_path: Optional[Path] = None,
	dpi: int = 200,
	gridsize: int = 10,
	image_size: float = 0.03,
	crop_size: int = 36,
	font_size: float = 10.0,
	line_width: float = 1.0,
) -> List[int]:
	"""Render a grid of ternary plots with actual images at each point."""

	if not slices:
		raise ValueError("No ternary slices to plot.")

	order = list(range(len(slices)))

	if max_plots is not None:
		order = order[:max_plots]

	total_plots = len(order)
	if total_plots == 0:
		raise ValueError("No plots selected after applying max_plots filter.")

	if nrow is None or ncol is None:
		ncol = ncol or math.ceil(math.sqrt(total_plots))
		nrow = nrow or math.ceil(total_plots / ncol)

	fig = plt.figure(figsize=fig_size)

	for pos, idx in enumerate(order, start=1):
		ax = fig.add_subplot(nrow, ncol, pos, projection="ternary")
		plot_single_ternary_with_images(
			slices[idx],
			ax,
			gridsize=gridsize,
			image_size=image_size,
			crop_size=crop_size,
			font_size=font_size,
			line_width=line_width,
		)

	fig.tight_layout()
	fig.subplots_adjust(wspace=0.5, hspace=0.5)

	if save_path:
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
		print(f"Saved ternary grid to {save_path}")
	else:
		plt.show()

	plt.close(fig)
	return order


def save_individual_ternary_images(
	slices: List[TernarySlice],
	output_dir: Path,
	*,
	fig_size: Tuple[float, float] = (8.0, 8.0),
	dpi: int = 300,
	gridsize: int = 10,
	image_size: float = 0.03,
	crop_size: int = 36,
	font_size: float = 10.0,
	line_width: float = 1.0,
	element_names: Optional[List[str]] = None,
) -> List[dict]:
	"""Save each ternary plot as an individual PNG file.
	
	Returns a list of metadata dictionaries with file paths and composition info.
	"""
	import time
	
	if not slices:
		raise ValueError("No ternary slices to plot.")
	
	output_dir.mkdir(parents=True, exist_ok=True)
	
	metadata = []
	total = len(slices)
	
	print(f"Saving {total} individual ternary plots to {output_dir}...")
	start_time = time.time()
	
	for idx, slice_data in enumerate(slices):
		if idx % 50 == 0 and idx > 0:
			elapsed = time.time() - start_time
			rate = idx / elapsed
			remaining = (total - idx) / rate
			print(f"  Progress: {idx}/{total} ({100*idx//total}%) - {elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining")
		
		# Create figure for single plot
		fig = plt.figure(figsize=fig_size)
		ax = fig.add_subplot(111, projection="ternary")
		
		plot_single_ternary_with_images(
			slice_data,
			ax,
			gridsize=gridsize,
			image_size=image_size,
			crop_size=crop_size,
			font_size=font_size,
			line_width=line_width,
		)
		
		# Generate filename
		combination = slice_data.combination
		filename = f"ternary_{idx:04d}_{'_'.join(map(str, combination))}.png"
		filepath = output_dir / filename
		
		# Save with tight layout
		fig.tight_layout()
		fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
		plt.close(fig)
		
		# Store metadata
		if element_names:
			element_strs = [element_names[i] for i in combination]
		else:
			element_strs = [f"Element_{i}" for i in combination]
		
		metadata.append({
			'index': idx,
			'filename': filename,
			'filepath': str(filepath),
			'combination': combination,
			'element_indices': ','.join(map(str, combination)),
			'element_1': element_strs[0],
			'element_2': element_strs[1],
			'element_3': element_strs[2],
			'num_samples': len(slice_data.selection_indices),
		})
	
	total_time = time.time() - start_time
	print(f"✓ Saved {total} plots in {total_time:.1f}s ({total_time/total:.2f}s per plot)")
	
	# Save metadata to CSV
	import csv
	metadata_path = output_dir / 'plots_metadata.csv'
	with open(metadata_path, 'w', newline='') as f:
		if metadata:
			writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
			writer.writeheader()
			writer.writerows(metadata)
	
	print(f"✓ Saved metadata to {metadata_path}")
	
	return metadata


def save_individual_ternary_images_incremental(
	slices: List[TernarySlice],
	output_dir: Path,
	image_provider: ImageColorProvider,
	*,
	fig_size: Tuple[float, float] = (8.0, 8.0),
	dpi: int = 300,
	gridsize: int = 10,
	image_size: float = 0.03,
	crop_size: int = 36,
	font_size: float = 10.0,
	line_width: float = 1.0,
	element_names: Optional[List[str]] = None,
	batch_size: int = 100,
	image_load_batch_size: int = 500,
	parallel: bool = True,
	num_workers: int = 4,
) -> List[dict]:
	"""Save ternary plots incrementally to avoid loading all images into memory.
	
	Processes slices in batches, loading images, saving plots, then clearing memory.
	"""
	import time
	import gc
	
	if not slices:
		raise ValueError("No ternary slices to plot.")
	
	output_dir.mkdir(parents=True, exist_ok=True)
	
	metadata = []
	total = len(slices)
	
	print(f"Saving {total} individual ternary plots incrementally (batch size: {batch_size})...")
	start_time = time.time()
	
	# Process in batches
	for batch_start in range(0, total, batch_size):
		batch_end = min(batch_start + batch_size, total)
		batch_slices = slices[batch_start:batch_end]
		batch_num = (batch_start // batch_size) + 1
		total_batches = (total + batch_size - 1) // batch_size
		
		print(f"\n=== Batch {batch_num}/{total_batches}: Processing slices {batch_start} to {batch_end-1} ===")
		
		# Load images for this batch
		batch_indices = np.concatenate([s.selection_indices for s in batch_slices])
		print(f"  Loading {len(batch_indices):,} images for this batch...")
		
		batch_images = image_provider.fetch_images(
			batch_indices,
			batch_size=image_load_batch_size,
			parallel=parallel,
			num_workers=num_workers
		)
		
		# Distribute images to slices
		print(f"  Distributing images to {len(batch_slices)} slices...")
		current_pos = 0
		for slice_data in batch_slices:
			num_images = len(slice_data.selection_indices)
			slice_data.images = batch_images[current_pos:current_pos + num_images]
			current_pos += num_images
		
		# Save plots for this batch
		print(f"  Saving {len(batch_slices)} plots...")
		for i, slice_data in enumerate(batch_slices):
			idx = batch_start + i
			
			# Create figure for single plot
			fig = plt.figure(figsize=fig_size)
			ax = fig.add_subplot(111, projection="ternary")
			
			plot_single_ternary_with_images(
				slice_data,
				ax,
				gridsize=gridsize,
				image_size=image_size,
				crop_size=crop_size,
				font_size=font_size,
				line_width=line_width,
			)
			
			# Generate filename
			combination = slice_data.combination
			filename = f"ternary_{idx:04d}_{'_'.join(map(str, combination))}.png"
			filepath = output_dir / filename
			
			# Save with tight layout
			fig.tight_layout()
			fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
			plt.close(fig)
			
			# Store metadata
			if element_names:
				element_strs = [element_names[j] for j in combination]
			else:
				element_strs = [f"Element_{j}" for j in combination]
			
			metadata.append({
				'index': idx,
				'filename': filename,
				'filepath': str(filepath),
				'combination': combination,
				'element_indices': ','.join(map(str, combination)),
				'element_1': element_strs[0],
				'element_2': element_strs[1],
				'element_3': element_strs[2],
				'num_samples': len(slice_data.selection_indices),
			})
		
		# Clear images from memory
		del batch_images
		for slice_data in batch_slices:
			slice_data.images = None
		gc.collect()
		
		elapsed = time.time() - start_time
		plots_done = batch_end
		rate = plots_done / elapsed
		remaining = (total - plots_done) / rate if plots_done < total else 0
		print(f"  ✓ Batch {batch_num} complete: {plots_done}/{total} plots saved ({100*plots_done//total}%)")
		print(f"    Time: {elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining")
	
	total_time = time.time() - start_time
	print(f"\n✓ Saved all {total} plots in {total_time:.1f}s ({total_time/total:.2f}s per plot)")
	
	# Save metadata to CSV
	import csv
	metadata_path = output_dir / 'plots_metadata.csv'
	with open(metadata_path, 'w', newline='') as f:
		if metadata:
			writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
			writer.writeheader()
			writer.writerows(metadata)
	
	print(f"✓ Saved metadata to {metadata_path}")
	
	return metadata


def plot_all_ternary_with_images_multipage(
	slices: List[TernarySlice],
	*,
	fig_size: Tuple[float, float] = (30.0, 30.0),
	nrow: int = 6,
	ncol: int = 8,
	plots_per_page: int = 48,
	save_path: Path,
	dpi: int = 200,
	gridsize: int = 10,
	image_size: float = 0.03,
	crop_size: int = 36,
	font_size: float = 10.0,
	line_width: float = 1.0,
	element_names: Optional[List[str]] = None,
) -> int:
	"""Render ternary plots across multiple PDF pages.
	
	Args:
		slices: List of ternary slices to plot
		fig_size: Size of each page in inches
		nrow: Number of rows per page
		ncol: Number of columns per page
		plots_per_page: Number of plots per page (should equal nrow * ncol)
		save_path: Path to save the multi-page PDF
		dpi: Resolution for the PDF
		gridsize: Grid size for hexbin background
		image_size: Size of displayed images
		crop_size: Size to crop from center of each image
		font_size: Font size for axis labels
		line_width: Line width for axes and ticks
		element_names: List of element names for the index file
		
	Returns:
		Total number of plots created
	"""
	if not slices:
		raise ValueError("No ternary slices to plot.")
	
	if not save_path.suffix.lower() == '.pdf':
		raise ValueError("Multi-page output requires PDF format (.pdf extension)")
	
	total_plots = len(slices)
	num_pages = math.ceil(total_plots / plots_per_page)
	
	print(f"Creating {num_pages} pages with up to {plots_per_page} plots per page ({total_plots} total plots)...")
	
	save_path.parent.mkdir(parents=True, exist_ok=True)
	
	# Pre-load ALL images for ALL slices at once (major speedup!)
	print(f"Pre-loading images for all {total_plots} ternary combinations...")
	all_images_loaded = sum(1 for s in slices if s.images is not None)
	if all_images_loaded == 0:
		raise ValueError("No images found in slices. Make sure images are loaded before calling this function.")
	print(f"  All {total_plots} combinations already have images loaded.")
	
	# Create index file with composition information
	index_path = save_path.with_suffix('.csv')
	index_rows = []
	
	with PdfPages(save_path) as pdf:
		for page_num in range(num_pages):
			start_idx = page_num * plots_per_page
			end_idx = min(start_idx + plots_per_page, total_plots)
			page_slices = slices[start_idx:end_idx]
			
			import time
			page_start_time = time.time()
			print(f"  Page {page_num + 1}/{num_pages}: Plotting {len(page_slices)} diagrams...")
			
			fig = plt.figure(figsize=fig_size)
			
			# Add page number text at the top
			fig.suptitle(f'Page {page_num + 1}/{num_pages}', fontsize=font_size * 2, y=0.98)
			
			for pos, slice_idx in enumerate(range(len(page_slices)), start=1):
				# Calculate row and column (1-indexed)
				row_num = ((pos - 1) // ncol) + 1
				col_num = ((pos - 1) % ncol) + 1
				
				ax = fig.add_subplot(nrow, ncol, pos, projection="ternary")
				
				# Add row/column label to the plot
				plot_label = f"R{row_num}C{col_num}"
				ax.text(0.5, 1.05, plot_label, transform=ax.transAxes,
				        fontsize=font_size * 1.2, ha='center', va='bottom',
				        fontweight='bold')
				
				plot_single_ternary_with_images(
					page_slices[slice_idx],
					ax,
					gridsize=gridsize,
					image_size=image_size,
					crop_size=crop_size,
					font_size=font_size,
					line_width=line_width,
				)
				
				# Record index information
				slice_data = page_slices[slice_idx]
				combination = slice_data.combination
				
				# Get element names if provided
				if element_names:
					element_strs = [element_names[idx] for idx in combination]
				else:
					element_strs = [f"Element_{idx}" for idx in combination]
				
				index_rows.append({
					'page': page_num + 1,
					'row': row_num,
					'column': col_num,
					'position_label': plot_label,
					'element_indices': ','.join(map(str, combination)),
					'element_1': element_strs[0],
					'element_2': element_strs[1],
					'element_3': element_strs[2],
					'num_samples': len(slice_data.selection_indices),
				})
			
			fig.subplots_adjust(wspace=0.3, hspace=0.4, top=0.96, left=0.02, right=0.98, bottom=0.02)
			
			# Save without bbox_inches="tight" for much faster rendering
			pdf.savefig(fig, dpi=dpi)
			plt.close(fig)
			
			page_time = time.time() - page_start_time
			print(f"    Page {page_num + 1} completed in {page_time:.1f}s ({page_time/len(page_slices):.2f}s per plot)")
	
	# Save index file
	import csv
	with open(index_path, 'w', newline='') as csvfile:
		fieldnames = ['page', 'row', 'column', 'position_label', 'element_indices', 
		              'element_1', 'element_2', 'element_3', 'num_samples']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(index_rows)
	
	print(f"Saved {num_pages}-page PDF to {save_path}")
	print(f"Saved index file to {index_path}")
	return total_plots


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Visualise ternary compositions colored by image properties.")
	parser.add_argument(
		"--compositions",
		required=True,
		help="Path to compositions array (.npy/.npz/.h5). Use path::dataset for npz/h5 entries.",
	)
	parser.add_argument("--compositions-key", help="Dataset/key within compositions npz/h5 file.")
	parser.add_argument("--element-names", type=Path, help="JSON or text file listing element names (one per line).")
	parser.add_argument(
		"--parse-json-compositions",
		action="store_true",
		help="Parse compositions as JSON strings (e.g., from generated samples) and convert to numeric arrays.",
	)
	parser.add_argument("--reference-compositions", type=Path, help="JSON list of reference compositions to prioritise.")
	parser.add_argument("--top-n", type=int, default=6, help="Number of most popular combinations to plot.")
	parser.add_argument("--num-elements", type=int, default=3, help="Elements per combination (default: 3).")
	parser.add_argument("--min-elements-present", type=int, default=2, help="Minimum elements from combination required in a sample.")
	parser.add_argument(
		"--color-mode",
		choices=COLOR_MODES_ALL,
		default=COLOR_MODE_MEAN_RGB,
		help="Colouring strategy: mean_rgb (default), dominant_hue, or grayscale.",
	)
	parser.add_argument(
		"--plot-mode",
		choices=["colors", "images"],
		default="colors",
		help="Plot mode: 'colors' uses averaged colors (default), 'images' displays cropped images at each point.",
	)
	parser.add_argument(
		"--image-source",
		required=True,
		help="Image dataset (.npy/.npz/.h5). Required for visualization.",
	)
	parser.add_argument("--image-key", help="Dataset/key within image npz/h5 file.")
	parser.add_argument(
		"--color-crop-size",
		type=int,
		default=0,
		help="Optional centre crop size (pixels) applied before computing image colours.",
	)
	parser.add_argument(
		"--image-crop-size",
		type=int,
		default=36,
		help="Crop size (pixels) for images when using --plot-mode images (default: 36).",
	)
	parser.add_argument(
		"--image-display-size",
		type=float,
		default=0.03,
		help="Display size for images in plot coordinates when using --plot-mode images (default: 0.03).",
	)
	parser.add_argument(
		"--parallel-loading",
		action="store_true",
		help="Enable parallel image loading for faster HDF5 access (recommended for large datasets).",
	)
	parser.add_argument(
		"--num-workers",
		type=int,
		default=4,
		help="Number of parallel workers for image loading (default: 4). Only used with --parallel-loading.",
	)
	parser.add_argument(
		"--hue-bins",
		type=int,
		default=60,
		help="Histogram bins for dominant hue computation.",
	)
	parser.add_argument(
		"--color-alpha",
		type=float,
		default=0.85,
		help="Scatter alpha when plotting colors.",
	)
	parser.add_argument("--fig-width", type=float, default=30.0, help="Figure width in inches.")
	parser.add_argument("--fig-height", type=float, default=30.0, help="Figure height in inches.")
	parser.add_argument("--nrow", type=int, help="Number of subplot rows (auto if omitted).")
	parser.add_argument("--ncol", type=int, help="Number of subplot columns (auto if omitted).")
	parser.add_argument("--max-plots", type=int, help="Limit the number of ternary plots shown.")
	parser.add_argument("--max-samples-per-plot", type=int, help="Limit samples per ternary plot (for large datasets).")
	parser.add_argument("--min-samples-per-combination", type=int, default=1, help="Minimum samples required for a combination to be included (default: 1).")
	parser.add_argument("--sample-size-for-combinations", type=int, help="Sample size for finding popular combinations (default: 50000). Increase for better coverage of large datasets.")
	parser.add_argument("--save", type=Path, help="Path to save the resulting figure instead of displaying it.")
	parser.add_argument("--dpi", type=int, default=200, help="Figure DPI when saving.")
	parser.add_argument("--gridsize", type=int, default=10, help="Hexbin grid size per plot.")
	parser.add_argument("--font-size", type=float, default=10.0, help="Font size for axis labels (default: 10).")
	parser.add_argument("--line-width", type=float, default=1.0, help="Line width for axes and ticks (default: 1.0).")
	parser.add_argument(
		"--multi-page",
		action="store_true",
		help="Create multi-page PDF with plots distributed across pages (requires PDF output).",
	)
	parser.add_argument(
		"--plots-per-page",
		type=int,
		help="Number of plots per page for multi-page PDF (default: nrow * ncol).",
	)
	parser.add_argument(
		"--save-individual-plots",
		action="store_true",
		help="Save each ternary plot as an individual PNG file instead of creating a combined PDF.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		help="Output directory for individual plot PNGs (used with --save-individual-plots).",
	)
	parser.add_argument(
		"--incremental-save",
		action="store_true",
		help="Save plots incrementally and clear memory periodically (recommended for large datasets with --save-individual-plots).",
	)
	parser.add_argument(
		"--incremental-batch-size",
		type=int,
		default=100,
		help="Number of plots to process before clearing memory when using --incremental-save (default: 100).",
	)
	parser.add_argument(
		"--image-load-batch-size",
		type=int,
		default=500,
		help="Batch size for loading images from HDF5 (default: 500). Larger = faster but more memory.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	compositions_path, compositions_key = split_path_and_key(args.compositions, args.compositions_key)
	
	# Load element names first if provided (needed for JSON parsing)
	element_names_list = None
	if args.element_names:
		if args.parse_json_compositions:
			element_names_list = load_element_names(args.element_names, expected=None)  # type: ignore
		else:
			temp_comp = load_array(compositions_path, compositions_key)
			num_elements = temp_comp.shape[1] if temp_comp.ndim == 2 else temp_comp.shape[0]
			element_names_list = load_element_names(args.element_names, expected=num_elements)
	
	print("Loading compositions...")
	compositions = load_array(
		compositions_path, 
		compositions_key,
		parse_json_compositions=args.parse_json_compositions,
		element_names=element_names_list
	)
	print(f"Loaded {len(compositions)} compositions")

	if element_names_list is None:
		element_names_list = load_element_names(args.element_names, compositions.shape[1]) if args.element_names else None
	
	image_path, image_key = split_path_and_key(args.image_source, args.image_key)
	reference = load_reference_compositions(args.reference_compositions)

	print("Finding ternary combinations...")
	ternary_slices = reduce_compositions(
		compositions,
		reference_compositions=reference,
		element_names=element_names_list,
		num_elements=args.num_elements,
		min_elements_present=args.min_elements_present,
		top_n=args.top_n,
		max_samples_per_combination=args.max_samples_per_plot,
		min_samples_per_combination=args.min_samples_per_combination,
		sample_size=args.sample_size_for_combinations,
	)

	# Load images or compute colors depending on plot mode
	# Skip loading if using incremental save mode (loads on-demand)
	if args.plot_mode == "images" and not (args.save_individual_plots and args.incremental_save):
		print(f"Loading images for plotting (crop size: {args.image_crop_size})...")
		total_images_to_load = sum(len(s.selection_indices) for s in ternary_slices)
		print(f"Total images to load across {len(ternary_slices)} combinations: {total_images_to_load:,}")
		
		# Collect all indices and create a mapping to slices
		print(f"Collecting all {total_images_to_load:,} unique indices for batch loading...")
		all_indices = np.concatenate([s.selection_indices for s in ternary_slices])
		
		# Create mapping from global index to (slice_idx, local_idx)
		index_to_slice_map = {}
		for slice_idx, slice_data in enumerate(ternary_slices):
			for local_idx, global_idx in enumerate(slice_data.selection_indices):
				index_to_slice_map[int(global_idx)] = (slice_idx, local_idx)
		
		# Load ALL images in one batch
		with ImageColorProvider(
			image_path,
			image_key,
			args.color_mode,  # Not used for image mode, but required for initialization
			crop_size=args.image_crop_size,
			hue_bins=args.hue_bins,
		) as provider:
			print(f"Loading all {len(all_indices):,} images in a single batch...")
			all_images = provider.fetch_images(
				all_indices,
				parallel=args.parallel_loading,
				num_workers=args.num_workers
			)
		
		# Distribute images back to slices
		print(f"Distributing {len(all_images):,} images to {len(ternary_slices)} combinations...")
		for slice_idx, slice_data in enumerate(ternary_slices):
			num_images = len(slice_data.selection_indices)
			slice_data.images = np.empty((num_images,) + all_images.shape[1:], dtype=all_images.dtype)
			
			for local_idx, global_idx in enumerate(slice_data.selection_indices):
				# Find where this global_idx appears in all_indices
				position = np.where(all_indices == global_idx)[0][0]
				slice_data.images[local_idx] = all_images[position]
			
			if (slice_idx + 1) % 50 == 0 or slice_idx == 0 or slice_idx == len(ternary_slices) - 1:
				print(f"  Distributed images to {slice_idx + 1}/{len(ternary_slices)} combinations")
		
		print(f"✓ All {total_images_to_load:,} images loaded and distributed successfully!")
	elif args.plot_mode == "colors":
		# Color mode - compute averaged colors
		print(f"Computing colors using '{args.color_mode}' mode...")
		total_samples_to_process = sum(len(s.selection_indices) for s in ternary_slices)
		samples_processed = 0
		print(f"Total samples to process across {len(ternary_slices)} combinations: {total_samples_to_process:,}")
		
		with ImageColorProvider(
			image_path,
			image_key,
			args.color_mode,
			crop_size=args.color_crop_size,
			hue_bins=args.hue_bins,
		) as provider:
			for combo_idx, slice_data in enumerate(ternary_slices, 1):
				num_samples = len(slice_data.selection_indices)
				print(f"  [{combo_idx}/{len(ternary_slices)}] Computing colors for {num_samples} samples from combination {slice_data.combination}...")
				
				colors = provider.fetch(slice_data.selection_indices)
				if colors.shape[0] != slice_data.reduced_compositions.shape[0]:
					raise ValueError(
						f"Image dataset yielded {colors.shape[0]} colours but slice has {slice_data.reduced_compositions.shape[0]} samples."
					)
				slice_data.colors = colors
				
				samples_processed += num_samples
				progress_pct = 100 * samples_processed / total_samples_to_process
				print(f"  Progress: {samples_processed:,}/{total_samples_to_process:,} samples processed ({progress_pct:.1f}%)")
		
		print(f"✓ All {samples_processed:,} colors computed successfully!")

	matplotlib_defaults = plt.rcParams
	matplotlib_defaults["axes.grid"] = False

	# Handle individual plot saving mode
	if args.save_individual_plots:
		if args.plot_mode != "images":
			raise ValueError("--save-individual-plots currently only supports --plot-mode images")
		
		output_dir = args.output_dir or Path("ternary_plots")
		
		if args.incremental_save:
			# Incremental mode: load and save in batches to avoid memory issues
			print(f"Using incremental save mode (batch size: {args.incremental_batch_size})...")
			
			with ImageColorProvider(
				image_path,
				image_key,
				args.color_mode,  # Not used for image mode, but required for initialization
				crop_size=args.image_crop_size,
				hue_bins=args.hue_bins,
			) as provider:
				metadata = save_individual_ternary_images_incremental(
					ternary_slices,
					output_dir,
					provider,
					fig_size=(8.0, 8.0),
					dpi=args.dpi,
					gridsize=args.gridsize,
					image_size=args.image_display_size,
					crop_size=args.image_crop_size,
					font_size=args.font_size,
					line_width=args.line_width,
					element_names=element_names_list,
					batch_size=args.incremental_batch_size,
					image_load_batch_size=args.image_load_batch_size,
					parallel=args.parallel_loading,
					num_workers=args.num_workers,
				)
		else:
			# Standard mode: load all images first, then save (requires images to be pre-loaded)
			print(f"Plotting {len(ternary_slices)} ternary visualizations as individual PNG files...")
			metadata = save_individual_ternary_images(
				ternary_slices,
				output_dir,
				fig_size=(8.0, 8.0),
				dpi=args.dpi,
				gridsize=args.gridsize,
				image_size=args.image_display_size,
				crop_size=args.image_crop_size,
				font_size=args.font_size,
				line_width=args.line_width,
				element_names=element_names_list,
			)
		
		print(f"Done! Saved {len(metadata)} individual ternary plot images to {output_dir}")
		return
	
	print(f"Plotting {len(ternary_slices)} ternary visualizations...")

	# Determine plots per page for multi-page mode
	if args.multi_page:
		if not args.save or not args.save.suffix.lower() == '.pdf':
			raise ValueError("Multi-page mode (--multi-page) requires PDF output (--save filename.pdf)")
		
		if args.nrow is None or args.ncol is None:
			raise ValueError("Multi-page mode requires explicit --nrow and --ncol arguments")
		
		plots_per_page = args.plots_per_page or (args.nrow * args.ncol)
	
	if args.plot_mode == "images":
		if args.multi_page:
			total_plots = plot_all_ternary_with_images_multipage(
				ternary_slices,
				fig_size=(args.fig_width, args.fig_height),
				nrow=args.nrow,
				ncol=args.ncol,
				plots_per_page=plots_per_page,
				save_path=args.save,
				dpi=args.dpi,
				gridsize=args.gridsize,
				image_size=args.image_display_size,
				crop_size=args.image_crop_size,
				font_size=args.font_size,
				line_width=args.line_width,
				element_names=element_names_list,
			)
			order = list(range(total_plots))
		else:
			order = plot_all_ternary_with_images(
				ternary_slices,
				fig_size=(args.fig_width, args.fig_height),
				nrow=args.nrow,
				ncol=args.ncol,
				max_plots=args.max_plots,
				save_path=args.save,
				dpi=args.dpi,
				gridsize=args.gridsize,
				image_size=args.image_display_size,
				crop_size=args.image_crop_size,
				font_size=args.font_size,
				line_width=args.line_width,
			)
	else:
		if args.multi_page:
			raise ValueError("Multi-page mode currently only supports --plot-mode images")
		
		order = plot_all_ternary(
			ternary_slices,
			fig_size=(args.fig_width, args.fig_height),
			nrow=args.nrow,
			ncol=args.ncol,
			max_plots=args.max_plots,
			save_path=args.save,
			dpi=args.dpi,
			gridsize=args.gridsize,
			color_alpha=args.color_alpha,
			font_size=args.font_size,
			line_width=args.line_width,
		)

	print(f"Done! Plotted {len(order)} ternary diagrams.")


if __name__ == "__main__":
	main()