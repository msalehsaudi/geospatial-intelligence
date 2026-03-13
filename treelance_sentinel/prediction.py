import os
import glob
import geopandas as gpd
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from loguru import logger
from treelance_sentinel.utils import setup_logger, timer_decorator
from typing import Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import rasterio
from rasterio import features
from rasterio.transform import from_origin
from treelance_sentinel.training import UNetClassifier, DeepUNetClassifier  # Import models from training module
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import shutil
import json
import signal
import sys
import atexit
from tqdm import tqdm
import inspect
import threading
import subprocess
import time

# Module-level flag to ensure retraining happens only once per workflow run
_retraining_already_occurred = False


def _get_safe_torch_load_kwargs():
    """Return kwargs and ensure legacy globals are allow-listed for torch.load."""
    try:
        torch.serialization.add_safe_globals(
            [
                np.core.multiarray._reconstruct,  # legacy numpy tensor wrapper
                np.ndarray,  # allow storing raw numpy arrays in checkpoints
                np.dtype,  # dtype metadata saved in older checkpoints
            ]
        )
    except AttributeError:
        # Older torch versions do not expose add_safe_globals; ignore
        pass

    load_kwargs = {}
    try:
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = False
    except (ValueError, TypeError):
        # Built-in/compiled function; signature introspection may fail. Fall back silently.
        pass
    return load_kwargs


def _torch_load(path: str, map_location=None):
    """Load a checkpoint with legacy-safe settings."""
    load_kwargs = _get_safe_torch_load_kwargs()
    if map_location is not None:
        load_kwargs["map_location"] = map_location

    try:
        return torch.load(path, **load_kwargs)
    except TypeError:
        # Older torch versions may not accept weights_only. Retry without it.
        load_kwargs.pop("weights_only", None)
        if map_location is not None:
            load_kwargs["map_location"] = map_location
        return torch.load(path, **load_kwargs)

# Configure CUDA allocator to reduce fragmentation (no-op on CPU-only)
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Global variables for cleanup
_global_executor = None
_global_model = None
_global_scaler = None

# GPU monitoring thread
_gpu_monitor_thread = None
_gpu_monitor_stop_event = None
_gpu_monitor_log_file = None


def _gpu_monitor_worker(stop_event, log_file_path, interval=5):
    """Background worker thread that periodically logs nvidia-smi output."""
    try:
        # Create a separate logger for GPU monitoring
        from loguru import logger as gpu_logger
        import sys
        
        # Remove default handler and add file handler
        gpu_logger.remove()
        gpu_logger.add(
            log_file_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level="INFO",
            rotation="100 MB",
            retention="7 days",
        )
        gpu_logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>GPU</cyan> | {message}",
            level="INFO",
            filter=lambda record: "GPU" in record["message"] or "nvidia-smi" in record["message"],
        )
        
        gpu_logger.info("=" * 80)
        gpu_logger.info("GPU Monitoring Started (nvidia-smi)")
        gpu_logger.info(f"Logging to: {log_file_path}")
        gpu_logger.info(f"Update interval: {interval} seconds")
        gpu_logger.info("=" * 80)
        
        while not stop_event.is_set():
            try:
                # Run nvidia-smi with detailed output
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw', 
                     '--format=csv,noheader'],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                
                if result.returncode == 0:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    gpu_logger.info(f"\n[{timestamp}] GPU Status:")
                    gpu_logger.info("Index | Name | GPU% | Mem% | Used/Total | Temp | Power")
                    gpu_logger.info("-" * 80)
                    
                    for line in result.stdout.strip().splitlines():
                        if line.strip():
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) >= 8:
                                idx, name, gpu_util, mem_util, mem_used, mem_total, temp, power = parts[:8]
                                gpu_logger.info(f"  {idx} | {name[:20]:20} | {gpu_util:>5} | {mem_util:>5} | {mem_used}/{mem_total} | {temp:>4} | {power}")
                else:
                    gpu_logger.warning(f"nvidia-smi failed: {result.stderr.strip()}")
                
                # Also log process-specific GPU usage
                proc_result = subprocess.run(
                    ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', 
                     '--format=csv,noheader'],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                
                if proc_result.returncode == 0 and proc_result.stdout.strip():
                    gpu_logger.info("\nGPU Processes:")
                    gpu_logger.info("PID | Process Name | Memory (MB)")
                    gpu_logger.info("-" * 80)
                    for line in proc_result.stdout.strip().splitlines():
                        if line.strip():
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) >= 3:
                                pid, proc_name, mem = parts[:3]
                                gpu_logger.info(f"  {pid} | {proc_name[:30]:30} | {mem:>10}")
                
            except subprocess.TimeoutExpired:
                gpu_logger.warning("nvidia-smi timed out")
            except Exception as e:
                gpu_logger.warning(f"Error querying GPU: {e}")
            
            # Wait for interval or stop event
            stop_event.wait(interval)
            
    except Exception as e:
        logger.error(f"GPU monitor thread error: {e}")
    finally:
        gpu_logger.info("=" * 80)
        gpu_logger.info("GPU Monitoring Stopped")
        gpu_logger.info("=" * 80)


def start_gpu_monitoring(log_dir: str | None = None, interval: int = 5) -> str | None:
    """Start background GPU monitoring thread.
    
    Args:
        log_dir: Directory to write GPU log file. If None, uses current directory.
        interval: Update interval in seconds (default: 5)
    
    Returns:
        Path to GPU log file, or None if CUDA not available
    """
    global _gpu_monitor_thread, _gpu_monitor_stop_event, _gpu_monitor_log_file
    
    if not torch.cuda.is_available():
        logger.info("GPU monitoring: CUDA not available, skipping")
        return None
    
    # Stop existing monitor if running
    stop_gpu_monitoring()
    
    # Determine log file path
    if log_dir is None:
        log_dir = os.getcwd()
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"gpu_monitor_{timestamp}.log")
    _gpu_monitor_log_file = log_file_path
    
    # Create stop event and start thread
    _gpu_monitor_stop_event = threading.Event()
    _gpu_monitor_thread = threading.Thread(
        target=_gpu_monitor_worker,
        args=(_gpu_monitor_stop_event, log_file_path, interval),
        daemon=True,
        name="GPU-Monitor"
    )
    _gpu_monitor_thread.start()
    
    logger.info(f"GPU monitoring started: logging to {log_file_path} (interval: {interval}s)")
    logger.info(f"Monitor GPU usage in real-time: tail -f {log_file_path}")
    
    return log_file_path


def stop_gpu_monitoring():
    """Stop the background GPU monitoring thread."""
    global _gpu_monitor_thread, _gpu_monitor_stop_event
    
    if _gpu_monitor_stop_event is not None:
        _gpu_monitor_stop_event.set()
        _gpu_monitor_stop_event = None
    
    if _gpu_monitor_thread is not None:
        _gpu_monitor_thread.join(timeout=2)
        _gpu_monitor_thread = None
        logger.info("GPU monitoring stopped")


def cleanup_gpu():
    """Clean up GPU resources and CUDA context."""
    try:
        if torch.cuda.is_available():
            # Clear all CUDA caches
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Force garbage collection to release Python objects holding GPU memory
            import gc
            gc.collect()
            # Clear cache again after GC
            torch.cuda.empty_cache()
            logger.info("GPU resources cleaned up")
    except Exception as e:
        logger.warning(f"Error during GPU cleanup: {e}")

def cleanup_resources():
    """Clean up all resources including ThreadPoolExecutor, GPU, and monitoring."""
    # Stop GPU monitoring
    stop_gpu_monitoring()
    global _global_executor, _global_model, _global_scaler
    import gc
    
    try:
        # Clean up ThreadPoolExecutor
        if _global_executor is not None:
            _global_executor.shutdown(wait=True, cancel_futures=True)
            logger.info("ThreadPoolExecutor shutdown complete")
            _global_executor = None
        
        # Clean up model - move to CPU first if on GPU, then delete
        if _global_model is not None:
            try:
                # If model is on GPU, move to CPU first to free GPU memory
                if hasattr(_global_model, 'parameters'):
                    device = next(_global_model.parameters()).device
                    if device.type == 'cuda':
                        _global_model = _global_model.cpu()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Error moving model to CPU during cleanup: {e}")
            finally:
                del _global_model
                _global_model = None
        
        if _global_scaler is not None:
            del _global_scaler
            _global_scaler = None
        
        # Force garbage collection before GPU cleanup
        gc.collect()
        
        # Clean up GPU
        cleanup_gpu()
        
    except Exception as e:
        logger.warning(f"Error during resource cleanup: {e}")

def signal_handler(sig, frame):
    """Handle interrupt signals gracefully."""
    logger.info(f"Received signal {sig}, cleaning up resources...")
    cleanup_resources()
    sys.exit(0)


def gpu_preflight(config: dict) -> None:
    """Pre-flight GPU check and cleanup before heavy training.

    - Clears this process' CUDA cache
    - Logs other python GPU processes
    - Optionally kills or fails fast if other python GPU processes exist
    - Starts GPU monitoring if enabled
    """
    retrain_cfg = (config or {}).get("prediction", {}).get("retraining", {})
    fail_if_other = bool(retrain_cfg.get("fail_if_other_gpu_python_processes", True))
    kill_others = bool(retrain_cfg.get("kill_other_gpu_python_processes", False))
    kill_grace_seconds = float(retrain_cfg.get("kill_other_gpu_python_processes_grace_seconds", 8.0))
    enable_gpu_monitoring = bool(retrain_cfg.get("enable_gpu_monitoring", True))
    gpu_monitor_interval = int(retrain_cfg.get("gpu_monitor_interval", 5))

    if not torch.cuda.is_available():
        logger.info("GPU preflight: CUDA not available; skipping.")
        return

    # Clear our cached allocations first
    cleanup_gpu()
    
    # Start GPU monitoring if enabled
    if enable_gpu_monitoring:
        # Try to get log directory from config
        log_dir = None
        if config:
            base_dir = config.get("directories", {}).get("base_output_dir", "")
            if base_dir:
                logs_dir = config.get("directories", {}).get("logs", "logs")
                log_dir = os.path.join(base_dir, logs_dir)
            else:
                logs_dir = config.get("directories", {}).get("logs", "logs")
                log_dir = logs_dir
        
        start_gpu_monitoring(log_dir=log_dir, interval=gpu_monitor_interval)

    def _list_other_python_gpu_processes() -> list[tuple[int, str, int]]:
        import subprocess

        cmd = [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            logger.warning(f"GPU preflight: nvidia-smi failed: {proc.stderr.strip()}")
            return []

        current_pid = os.getpid()
        other_python: list[tuple[int, str, int]] = []
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            pid_str, name, mem_str = parts
            try:
                pid = int(pid_str)
                mem_mb = int(mem_str)
            except Exception:
                continue
            if pid == current_pid:
                continue
            if "python" in name.lower() and mem_mb > 0:
                other_python.append((pid, name, mem_mb))
        return other_python

    try:
        import subprocess
        import signal as _signal
        import time

        other_python = _list_other_python_gpu_processes()

        if not other_python:
            logger.info("GPU preflight: no other python GPU processes detected.")
            return

        logger.warning("GPU preflight: detected other python GPU processes:")
        for pid, name, mem_mb in other_python:
            logger.warning(f"- pid={pid} name={name} gpu_mem_mb={mem_mb}")

        if kill_others:
            logger.warning(
                "GPU preflight: kill_other_gpu_python_processes=true; attempting to terminate other python GPU processes..."
            )

            # First attempt: SIGTERM
            for pid, _, _ in other_python:
                try:
                    os.kill(pid, _signal.SIGTERM)
                except Exception as e:
                    logger.warning(f"Failed to SIGTERM pid={pid}: {e}")

            # Wait for processes to exit / release GPU
            deadline = time.time() + kill_grace_seconds
            while time.time() < deadline:
                time.sleep(0.5)
                still_using = _list_other_python_gpu_processes()
                if not still_using:
                    break

            still_using = _list_other_python_gpu_processes()
            if still_using:
                logger.warning("GPU preflight: some python GPU processes still present after SIGTERM; sending SIGKILL...")
                for pid, _, _ in still_using:
                    try:
                        os.kill(pid, _signal.SIGKILL)
                    except Exception as e:
                        logger.warning(f"Failed to SIGKILL pid={pid}: {e}")

                # Short wait after SIGKILL
                time.sleep(1.0)

            # Final verification
            remaining = _list_other_python_gpu_processes()
            if remaining:
                logger.warning("GPU preflight: GPU still used by python processes after kill attempts:")
                for pid, name, mem_mb in remaining:
                    logger.warning(f"- pid={pid} name={name} gpu_mem_mb={mem_mb}")
                if fail_if_other:
                    raise RuntimeError(
                        "GPU is currently used by other python processes even after kill attempts. "
                        "Please stop them manually and retry."
                    )
            else:
                logger.info("GPU preflight: ✅ cleared other python GPU processes.")

            cleanup_gpu()
            return

        if fail_if_other:
            raise RuntimeError(
                "GPU is currently used by other python processes. "
                "Please stop them (or set prediction.retraining.kill_other_gpu_python_processes=true)."
            )
    except Exception as e:
        logger.error(f"GPU preflight failed: {e}")
        raise

def _find_optimal_batch_size(model, feature_columns: list, default_batch_size: int, device) -> int:
    """
    Phase 1 Optimization: Find optimal batch size through profiling.
    
    Tests different batch sizes and selects the one that maximizes throughput
    while staying within GPU memory limits.
    """
    if not torch.cuda.is_available():
        return default_batch_size
    
    logger.info(f"Profiling optimal batch size (starting from {default_batch_size})...")
    
    # Clear GPU memory first
    cleanup_gpu()
    torch.cuda.empty_cache()
    
    # Get available GPU memory
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            free_memory_mb = float(result.stdout.strip())
        else:
            # Fallback
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            allocated_mem = torch.cuda.memory_allocated() / (1024**2)
            free_memory_mb = total_mem - allocated_mem
    except Exception:
        # Fallback
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        allocated_mem = torch.cuda.memory_allocated() / (1024**2)
        free_memory_mb = total_mem - allocated_mem
    
    logger.info(f"Available GPU memory: {free_memory_mb:.1f}MB")
    
    # Test batch sizes: start from default, then try larger if memory allows
    test_batch_sizes = [default_batch_size]
    
    # Add larger batch sizes if memory allows (based on profiling: ~10.7MB per sample)
    if free_memory_mb > 2000:  # If > 2GB free, try larger batches
        test_batch_sizes.extend([200, 300, 400, 500])
    elif free_memory_mb > 1500:  # If > 1.5GB free, try moderate batches
        test_batch_sizes.extend([200, 300])
    
    # Remove duplicates and sort
    test_batch_sizes = sorted(list(set(test_batch_sizes)))
    
    optimal_batch_size = default_batch_size
    best_throughput = 0
    
    # Create test data
    num_features = len(feature_columns)
    test_data = torch.randn(test_batch_sizes[-1], num_features).float().to(device)
    
    for bs in test_batch_sizes:
        if bs > len(test_data):
            continue
        
        try:
            # Clear cache before each test
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Test with a small sample
            test_batch = test_data[:bs]
            
            # Warmup
            with torch.no_grad():
                _ = model(test_batch)
            torch.cuda.synchronize()
            
            # Profile
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            with torch.no_grad():
                _ = model(test_batch)
            end_event.record()
            torch.cuda.synchronize()
            
            elapsed_ms = start_event.elapsed_time(end_event)
            throughput = (bs / elapsed_ms) * 1000  # samples per second
            
            # Check memory usage
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
            
            logger.debug(f"Batch size {bs}: {throughput:.0f} samples/s, peak memory: {peak_memory:.1f}MB")
            
            # Prefer larger batch sizes if throughput is similar and memory is acceptable
            if throughput > best_throughput and peak_memory < free_memory_mb * 0.7:
                optimal_batch_size = bs
                best_throughput = throughput
            
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logger.debug(f"Batch size {bs} caused OOM, stopping search")
                break
            else:
                raise
        except Exception as e:
            logger.warning(f"Error testing batch size {bs}: {e}")
            break
    
    # Cleanup
    del test_data
    cleanup_gpu()
    
    logger.info(f"Optimal batch size: {optimal_batch_size} (throughput: {best_throughput:.0f} samples/s)")
    
    return optimal_batch_size


# Register cleanup functions
atexit.register(cleanup_resources)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Class mapping definition
CLASS_NAMES = {
    0: 'grassland',
    1: 'tree',
    2: 'urban'
}

def load_pytorch_model(model_path: str):
    """Load a PyTorch model from the given path."""
    try:
        if isinstance(model_path, str) and model_path.startswith("s3://"):
            raise ValueError("S3 model paths are no longer supported. Please provide a local model checkpoint path.")

        actual_model_path = model_path
        
        # Ensure legacy globals are allowed for older checkpoints (PyTorch >=2.6 safety change)
        # Load the checkpoint - force GPU usage
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. GPU is required for model loading.")
        checkpoint = _torch_load(actual_model_path, map_location=None)  # None = use current device (GPU)
        
        # Get the model state dict and create a new model instance
        model_state_dict = checkpoint['model_state_dict']
        
        # Get input size from checkpoint - use the actual saved value
        input_size = checkpoint.get('input_size', 10)  # Default to 10 if not found
        num_classes = checkpoint.get('num_classes', 3)  # Default to 3 if not found
        
        # Infer actual input_size and num_classes from checkpoint weights if metadata is wrong
        try:
            if 'encoder.input_proj.0.weight' in checkpoint['model_state_dict']:
                actual_input_size = checkpoint['model_state_dict']['encoder.input_proj.0.weight'].shape[1]
                if actual_input_size != input_size:
                    logger.warning(f"Checkpoint metadata shows input_size={input_size}, but weights show {actual_input_size}. Using actual weight dimensions.")
                    input_size = actual_input_size
            
            # Check for DeepUNetClassifier output layer (final.8) first, then UNetClassifier (final.4)
            if 'final.8.weight' in checkpoint['model_state_dict']:
                actual_num_classes = checkpoint['model_state_dict']['final.8.weight'].shape[0]
                if actual_num_classes != num_classes:
                    logger.warning(f"Checkpoint metadata shows num_classes={num_classes}, but weights show {actual_num_classes}. Using actual weight dimensions.")
                    num_classes = actual_num_classes
            elif 'final.4.weight' in checkpoint['model_state_dict']:
                actual_num_classes = checkpoint['model_state_dict']['final.4.weight'].shape[0]
                if actual_num_classes != num_classes:
                    logger.warning(f"Checkpoint metadata shows num_classes={num_classes}, but weights show {actual_num_classes}. Using actual weight dimensions.")
                    num_classes = actual_num_classes
        except Exception:
            pass
        
        # Define feature columns based on actual input size from checkpoint
        feature_columns = [f'b{band_idx}_mean' for band_idx in range(1, input_size + 1)]
        
        # Log model checkpoint information
        logger.info("\n=== Model Checkpoint Information ===")
        logger.info(f"Checkpoint path: {model_path}")
        if isinstance(checkpoint, dict):
            if 'epoch' in checkpoint:
                logger.info(f"Training epoch: {checkpoint['epoch']}")
            if 'val_f1' in checkpoint:
                logger.info(f"Validation F1: {checkpoint['val_f1']:.4f}")
            if 'val_acc' in checkpoint:
                logger.info(f"Validation Accuracy: {checkpoint['val_acc']:.2f}%")
            if 'train_acc' in checkpoint:
                logger.info(f"Training Accuracy: {checkpoint['train_acc']:.2f}%")
            if 'val_loss' in checkpoint:
                logger.info(f"Validation Loss: {checkpoint['val_loss']:.4f}")
            if 'train_loss' in checkpoint:
                logger.info(f"Training Loss: {checkpoint['train_loss']:.4f}")
            logger.info(f"Input size: {input_size}")
            logger.info(f"Number of classes: {num_classes}")
        
        # Extract optional scaler stats and feature columns if present
        scaler_mean = checkpoint.get('scaler_mean')
        scaler_scale = checkpoint.get('scaler_scale')
        saved_feature_columns = checkpoint.get('feature_columns')
        # Harmonize saved_feature_columns with inferred input_size if present
        if isinstance(saved_feature_columns, list):
            if len(saved_feature_columns) != input_size:
                logger.warning(
                    f"Checkpoint feature_columns length ({len(saved_feature_columns)}) does not match "
                    f"input_size ({input_size}). Rebuilding feature columns to match weights."
                )
                saved_feature_columns = [f'b{band_idx}_mean' for band_idx in range(1, input_size + 1)]

        # Return lightweight metadata for feature columns; downstream uses GBDT joblib
        return {
            'input_size': input_size,
            'num_classes': num_classes,
            'feature_columns': saved_feature_columns or feature_columns,
            'scaler_mean': scaler_mean,
            'scaler_scale': scaler_scale,
        }
    except Exception as e:
        logger.error(f"Error loading PyTorch model: {e}")
        raise

def _list_current_models(model_dir, stem):
    """List all current models with their F1 scores."""
    try:
        model_files = []
        for file in os.listdir(model_dir):
            if file.startswith(stem) and file.endswith('.pt'):
                if '__f1-' in file:
                    try:
                        f1_part = file.split('__f1-')[1].split('__')[0]
                        f1_score = float(f1_part)
                        model_files.append((file, f1_score))
                    except (ValueError, IndexError):
                        continue
        
        if model_files:
            model_files.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"Current models in {model_dir}:")
            for filename, f1_score in model_files:
                logger.info(f"  📁 {filename} (F1: {f1_score:.4f})")
        else:
            logger.info(f"No existing models found with stem '{stem}' in {model_dir}")
            
    except Exception as e:
        logger.error(f"Error listing current models: {e}")


def _manage_model_versions(model_dir, stem, current_f1, current_model_path, max_models=5):
    """
    Manage model versions by keeping only the best performing models.
    
    Args:
        model_dir: Directory containing model files
        stem: Base name for the model files
        current_f1: F1 score of the current model
        current_model_path: Path to the current model file
        max_models: Maximum number of models to keep
    """
    try:
        # Find all existing model files with the same stem
        model_files = []
        for file in os.listdir(model_dir):
            if file.startswith(stem) and file.endswith('.pt'):
                # Extract F1 score from filename
                if '__f1-' in file:
                    try:
                        f1_part = file.split('__f1-')[1].split('__')[0]
                        f1_score = float(f1_part)
                        model_files.append((file, f1_score, os.path.join(model_dir, file)))
                    except (ValueError, IndexError):
                        # Skip files that don't match the expected naming pattern
                        continue
        
        # Sort by F1 score (descending)
        model_files.sort(key=lambda x: x[1], reverse=True)
        
        # Keep only the top max_models
        models_to_keep = model_files[:max_models]
        models_to_remove = model_files[max_models:]
        
        # Remove lower performing models
        for filename, f1_score, filepath in models_to_remove:
            try:
                os.remove(filepath)
                logger.info(f"Removed lower performing model: {filename} (F1: {f1_score:.4f})")
                
                # Also remove associated metadata file if it exists
                metadata_file = filepath.replace('.pt', '.metadata.json')
                if os.path.exists(metadata_file):
                    os.remove(metadata_file)
                    logger.info(f"Removed associated metadata: {os.path.basename(metadata_file)}")
                    
            except Exception as e:
                logger.warning(f"Failed to remove model file {filename}: {e}")
        
        # Log current model status
        logger.info(f"Model management complete. Keeping top {len(models_to_keep)} models:")
        for filename, f1_score, filepath in models_to_keep:
            status = "🆕 NEW" if filepath == current_model_path else "📁 EXISTING"
            logger.info(f"  {status} {filename} (F1: {f1_score:.4f})")
            
    except Exception as e:
        logger.error(f"Error in model management: {e}")


def retrain_model(model, low_conf_predictions, config):
    """Retrain the model using only labeled data from confidence files."""
    try:
        logger.info("Starting model retraining process...")
        gpu_preflight(config)
        
        # Get retraining parameters
        confidence_threshold = config.get('prediction', {}).get('retraining', {}).get('confidence_threshold', 0.8)
        
        # Log total labeled samples found
        total_labeled = len(low_conf_predictions)
        logger.info("=" * 80)
        logger.info("📊 LABELED DATA SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total labeled samples found: {total_labeled:,}")
        
        if total_labeled == 0:
            logger.warning("⚠️  No labeled data found! Cannot retrain model.")
            return model
        
        # Filter to keep only class IDs 0, 1, 2
        low_conf_predictions = low_conf_predictions[low_conf_predictions['class_id'].isin([0, 1, 2])]
        filtered_count = len(low_conf_predictions)
        
        if filtered_count < total_labeled:
            logger.warning(f"Filtered out {total_labeled - filtered_count} samples with invalid class_ids")
        
        logger.info(f"Valid labeled samples (class_id in [0,1,2]): {filtered_count:,}")
        class_dist = low_conf_predictions['class_id'].value_counts().sort_index()
        logger.info("Class distribution:")
        for class_id, count in class_dist.items():
            class_name = CLASS_NAMES.get(class_id, f"Unknown({class_id})")
            pct = (count / filtered_count * 100) if filtered_count > 0 else 0
            logger.info(f"  - {class_name} (class_id={class_id}): {count:,} samples ({pct:.1f}%)")
        logger.info("=" * 80)
        
        # Prepare features for labeled data
        # Ensure all expected feature columns exist; fill missing with zeros
        expected_cols = getattr(model, 'feature_columns', [])
        if not expected_cols:
            raise ValueError("Model does not have feature_columns defined for retraining")
        
        # Convert H3 cell ID string to numeric (if present and needed) - must match training conversion
        # H3 geo embeddings removed: hash-based H3 loses spatial proximity and creates
        # thousands of categorical values with no benefit. UTM zone_number already provides
        # spatial proximity. If H3 columns are present in data, they will be ignored.
        
        logger.info(f"=== RETRAINING FEATURE DEBUG ===")
        logger.info(f"Expected columns: {expected_cols}")
        logger.info(f"Available columns in data: {list(low_conf_predictions.columns)}")
        logger.info(f"Data shape: {low_conf_predictions.shape}")
        
        missing_cols = [c for c in expected_cols if c not in low_conf_predictions.columns]
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            for col in missing_cols:
                low_conf_predictions[col] = 0
                logger.info(f"Filled missing column {col} with zeros")
        else:
            logger.info("All expected columns present in data")
        
        # Use the exact expected column order
        X_labeled = low_conf_predictions[expected_cols].values
        y_labeled = low_conf_predictions['class_id'].values.astype(np.int64)  # Convert to int64
        
        logger.info(f"Final feature matrix shape: {X_labeled.shape}")
        logger.info(f"Sample feature values (first 3 rows, first 3 cols):")
        logger.info(f"{X_labeled[:3, :3]}")
        logger.info(f"Feature ranges: min={X_labeled.min(axis=0)[:3]}, max={X_labeled.max(axis=0)[:3]}")
        logger.info(f"=== END FEATURE DEBUG ===")
        
        # CRITICAL: Apply the same scaler that was used during original training
        # This ensures feature distributions match what the model expects
        # NOTE: 
        # - Categorical features (UTM first_digit, hemisphere_code) should NOT be scaled
        # - Numerical spatial features (utm_zone_number) should NOT be scaled (preserves spatial proximity)
        # - Continuous features (band means, etc.) ARE scaled
        # 
        # H3 geo embeddings removed: hash-based H3 loses spatial proximity and creates
        # thousands of categorical values with no benefit. UTM zone_number already provides
        # spatial proximity.
        categorical_features = []
        if 'utm_first_digit' in expected_cols:
            categorical_features.append('utm_first_digit')
        if 'utm_hemisphere_code' in expected_cols:
            categorical_features.append('utm_hemisphere_code')
        # Backward compatibility: old columns
        if 'utm_hemisphere' in expected_cols:
            categorical_features.append('utm_hemisphere')
        if 'utm_zone' in expected_cols:
            categorical_features.append('utm_zone')
        
        # Numerical spatial features (preserve spatial proximity, NOT standardized)
        numerical_spatial_features = []
        if 'utm_zone_number' in expected_cols:
            numerical_spatial_features.append('utm_zone_number')
        if 'h3_cell_id_regional_numeric' in expected_cols:
            categorical_features.append('h3_cell_id_regional_numeric')
        if 'h3_cell_id_local_numeric' in expected_cols:
            categorical_features.append('h3_cell_id_local_numeric')
        if 'h3_cell_id_numeric' in expected_cols:
            categorical_features.append('h3_cell_id_numeric')
        
        continuous_features = [col for col in expected_cols 
                              if col not in categorical_features and col not in numerical_spatial_features]
        
        if numerical_spatial_features:
            logger.info(f"Numerical spatial features (not scaled, preserve spatial proximity): {numerical_spatial_features}")
        
        logger.info(f"=== SCALER DEBUG ===")
        logger.info(f"Categorical features (not scaled): {categorical_features}")
        logger.info(f"Continuous features (scaled): {len(continuous_features)} features")
        logger.info(f"Model has scaler attribute: {hasattr(model, 'scaler')}")
        if hasattr(model, 'scaler'):
            logger.info(f"Model scaler is None: {model.scaler is None}")
            if model.scaler is not None:
                logger.info(f"Scaler type: {type(model.scaler)}")
                logger.info(f"Scaler mean shape: {model.scaler.mean_.shape if hasattr(model.scaler, 'mean_') else 'No mean'}")
                logger.info(f"Scaler scale shape: {model.scaler.scale_.shape if hasattr(model.scaler, 'scale_') else 'No scale'}")
        
        if hasattr(model, 'scaler') and model.scaler is not None:
            logger.info("Applying original training scaler to continuous features only...")
            logger.info(f"Feature stats BEFORE scaling: mean={X_labeled.mean(axis=0)[:3]}, std={X_labeled.std(axis=0)[:3]}")
            try:
                # Convert to DataFrame to preserve column names
                X_labeled_df = pd.DataFrame(X_labeled, columns=expected_cols)
                # Scale only continuous features
                if continuous_features:
                    X_labeled_df[continuous_features] = model.scaler.transform(X_labeled_df[continuous_features])
                X_labeled = X_labeled_df.values
                logger.info(f"Feature stats AFTER scaling: mean={X_labeled.mean(axis=0)[:3]}, std={X_labeled.std(axis=0)[:3]}")
                logger.info("✅ Scaler applied successfully!")
            except Exception as e:
                logger.error(f"❌ Scaler application failed: {e}")
                logger.warning("Proceeding with unscaled features - this will likely cause poor performance!")
        else:
            logger.warning("❌ No scaler found on model. Features may not match original training distribution!")
            logger.warning("This will likely cause poor retraining performance due to feature distribution mismatch!")
        logger.info(f"=== END SCALER DEBUG ===")
        
        # Check for and handle NaN values before SMOTE (SMOTE cannot handle NaN)
        logger.info("Checking for NaN values in features...")
        nan_mask = np.isnan(X_labeled).any(axis=1)
        nan_count = nan_mask.sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} rows with NaN values ({100*nan_count/len(X_labeled):.2f}% of data). Dropping these rows before SMOTE.")
            # Drop rows with NaN values
            X_labeled = X_labeled[~nan_mask]
            y_labeled = y_labeled[~nan_mask]
            logger.info(f"After NaN removal: {len(X_labeled)} samples remaining")
        else:
            logger.info("✅ No NaN values found in features")
        
        # Log class distribution before SMOTE
        logger.info("Class distribution before SMOTE:")
        class_dist = pd.Series(y_labeled).value_counts().sort_index()
        logger.info(class_dist)
        
        # Verify we only have classes 0, 1, 2
        if not set(class_dist.index).issubset({0, 1, 2}):
            raise ValueError("Found invalid class labels. Only classes 0, 1, 2 are allowed.")
        
        # Verify we have enough samples after NaN removal
        if len(X_labeled) == 0:
            raise ValueError("No valid samples remaining after NaN removal. Cannot proceed with retraining.")
        
        # Apply SMOTE for better class balance
        logger.info("Applying SMOTE to balance classes...")
        sm = SMOTE(random_state=42, k_neighbors=min(5, len(np.unique(y_labeled))-1))
        X_res, y_res = sm.fit_resample(X_labeled, y_labeled)
        logger.info(f"After SMOTE - Dataset shape: {X_res.shape}")
        logger.info("Class distribution after SMOTE:")
        logger.info(pd.Series(y_res).value_counts().sort_index())
        
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
        )
        
        # Set up training parameters - force GPU usage
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. GPU is required for model retraining.")
        device = torch.device('cuda')
        model = model.to(device)
        
        # Convert to PyTorch tensors and move to device
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.LongTensor(y_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.LongTensor(y_val).to(device)
        
        # Create datasets and dataloaders
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        
        # Smaller batch size for the more complex model
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        # Use weighted loss to handle class imbalance
        class_counts = np.bincount(y_res)
        class_weights = torch.FloatTensor(1.0 / class_counts).to(device)
        class_weights = class_weights / class_weights.sum()
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train the model
        logger.info("Training model on labeled data...")
        model.train()
        best_val_f1 = 0.0
        patience = 30  # Early stopping after 30 epochs
        patience_counter = 0
        best_epoch = 0
        best_val_labels = None
        best_val_preds = None
        
        # Get max epochs and patience from config
        max_epochs = config.get('prediction', {}).get('retraining', {}).get('epochs', 100)
        patience = config.get('prediction', {}).get('retraining', {}).get('patience', 20)
        
        for epoch in range(max_epochs):  # Max epochs from config
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]", leave=False)
            for features, labels in train_loader_tqdm:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                train_loader_tqdm.set_postfix(loss=loss.item())
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_val_labels = []
            all_val_preds = []
            
            with torch.no_grad():
                val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Val]", leave=False)
                for features, labels in val_loader_tqdm:
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    all_val_labels.extend(labels.cpu().numpy())
                    all_val_preds.extend(predicted.cpu().numpy())
            
            # Calculate metrics
            val_acc = 100 * val_correct / val_total
            val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
            
            # Log epoch results in one line
            logger.info(f"Epoch {epoch+1:3d} | Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {100*train_correct/train_total:5.2f}% | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:5.2f}% | Val F1: {val_f1:.4f}")
            
            # Save best model based on F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                patience_counter = 0
                # Store validation labels and predictions for confusion matrix
                best_val_labels = all_val_labels.copy() if isinstance(all_val_labels, np.ndarray) else np.array(all_val_labels)
                best_val_preds = all_val_preds.copy() if isinstance(all_val_preds, np.ndarray) else np.array(all_val_preds)
            else:
                patience_counter += 1
                logger.info(f"Epoch {epoch+1}: No improvement for {patience_counter} epochs (patience: {patience})")
            
            # Early stopping check
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement. Best F1: {best_val_f1:.4f} at epoch {best_epoch+1}")
                break
        
        # Training completed - log final summary
        if patience_counter >= patience:
            logger.info(f"Training stopped early at epoch {epoch+1}/{max_epochs} due to no improvement for {patience} epochs")
        else:
            logger.info(f"Training completed all {max_epochs} epochs")
        
        logger.info("=" * 80)
        logger.info("📈 TRAINING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Best validation F1: {best_val_f1:.4f} achieved at epoch {best_epoch+1}")
        logger.info(f"Total epochs trained: {epoch+1}/{max_epochs}")
        logger.info(f"Early stopping: {'Yes' if patience_counter >= patience else 'No'}")
        
        # Ensure best_val_labels and best_val_preds are set for confusion matrix
        if best_val_labels is None or best_val_preds is None:
            logger.warning("⚠️  best_val_labels or best_val_preds not set - confusion matrix will not be generated")
            # Try to get from last validation run
            if 'all_val_labels' in locals() and 'all_val_preds' in locals():
                best_val_labels = all_val_labels.copy() if isinstance(all_val_labels, np.ndarray) else np.array(all_val_labels)
                best_val_preds = all_val_preds.copy() if isinstance(all_val_preds, np.ndarray) else np.array(all_val_preds)
                logger.info("Using last validation run for confusion matrix")
        
        # Save the retrained model
        full_save_path = None
        save_path = config.get('prediction', {}).get('model_save_path')
        if save_path:
            # Get the base directory from config
            base_dir = config.get("directories", {}).get("base_output_dir")
            if not base_dir:
                raise ValueError("Base output directory not found in config")
            
            # Build a meaningful dynamic filename: <project>__<stem>__ts-<ts>__f1-<f1>__ft-<n>.pt
            ts = datetime.now().strftime("%Y%m%d")
            stem = os.path.splitext(os.path.basename(save_path))[0]
            
            # Extract project name from config (prefer AOI name, fallback to config filename or base_output_dir)
            project_name = None
            aoi = config.get("input", {}).get("aoi")
            if aoi and isinstance(aoi, str):
                # Extract project name from AOI (e.g., "prediction_ready_aoi.bayernwerk_eon_one_100" -> "bayernwerk_eon_one_100")
                if "." in aoi:
                    project_name = aoi.split(".")[-1]
                else:
                    project_name = aoi
                # Sanitize for filename (remove invalid chars)
                project_name = project_name.strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
            
            if not project_name:
                # Fallback: try to extract from base_output_dir
                base_dir = config.get("directories", {}).get("base_output_dir", "")
                if base_dir:
                    project_name = base_dir.rstrip("/").split("/")[-1]
            
            if not project_name:
                # Final fallback: use stem from model_save_path
                project_name = stem
            
            # Build filename with project name, timestamp, F1 score, and feature count
            # Format: <project>__<stem>__ts-<YYYYMMDD>__f1-<0.XXXX>__ft-<N>.pt
            dynamic_name = (
                f"{project_name}__{stem}__ts-{ts}__f1-{best_val_f1:.4f}__ft-{len(getattr(model,'feature_columns', []))}.pt"
            )
            
            logger.info(f"Model save path includes project name: {project_name}")
            model_dir = os.path.join(base_dir, os.path.dirname(save_path)).rstrip("/")
            full_save_path = os.path.join(model_dir, dynamic_name)
            
            # Collect scaler stats if present on model (ensures retraining output includes scaler)
            scaler_mean = None
            scaler_scale = None
            if hasattr(model, "scaler") and getattr(model, "scaler") is not None:
                try:
                    scaler_mean = getattr(model.scaler, "mean_", None)
                    scaler_scale = getattr(model.scaler, "scale_", None)
                except Exception:
                    pass
            
            # Save model with metadata (including scaler stats when available)
            model_data = {
                "model_state_dict": model.state_dict(),
                "input_size": len(model.feature_columns),
                "num_classes": len(CLASS_NAMES),
                "feature_columns": model.feature_columns,
                "timestamp": ts,
                "training_metadata": {
                    "confidence_threshold": config.get("prediction", {})
                    .get("retraining", {})
                    .get("confidence_threshold"),
                    "training_samples": len(y_labeled),
                    "training_class_distribution": class_dist.to_dict(),
                    "best_val_f1": best_val_f1,
                    "best_epoch": best_epoch + 1,
                    "total_epochs_trained": epoch + 1,
                    "early_stopping_triggered": patience_counter >= patience,
                    "patience": patience,
                    "max_epochs": max_epochs,
                },
            }
            if scaler_mean is not None and scaler_scale is not None:
                model_data["scaler_mean"] = np.array(scaler_mean)
                model_data["scaler_scale"] = np.array(scaler_scale)
            
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model_data, full_save_path)
            logger.info(f"Retrained model saved to {full_save_path}")
            
            # List current models before management (use project_name in stem)
            _list_current_models(model_dir, f"{project_name}__{stem}")
            
            # Manage model versions - keep only the best models (use project_name in stem for version management)
            max_models = config.get("prediction", {}).get("retraining", {}).get("max_models_to_keep", 5)
            _manage_model_versions(model_dir, f"{project_name}__{stem}", best_val_f1, full_save_path, max_models)
                
                # Save metadata separately for easier inspection (include project name)
            metadata_path = os.path.join(
                model_dir, f"{project_name}__{stem}__ts-{ts}__f1-{best_val_f1:.4f}.metadata.json"
            )
            with open(metadata_path, "w") as f:
                json.dump(
                    {
                        "timestamp": model_data["timestamp"],
                        "input_size": model_data["input_size"],
                        "num_classes": model_data["num_classes"],
                        "feature_columns": model_data["feature_columns"],
                        "training_metadata": model_data["training_metadata"],
                        "has_scaler": scaler_mean is not None and scaler_scale is not None,
                    },
                    f,
                    indent=2,
                )
                logger.info(f"Model metadata saved to {metadata_path}")
        
        # Generate and save confusion matrix for best model at the end
        logger.info("=" * 80)
        logger.info("📊 GENERATING CONFUSION MATRIX AND METRICS")
        logger.info("=" * 80)
        
        if best_val_labels is None or best_val_preds is None:
            logger.warning("⚠️  Cannot generate confusion matrix: best_val_labels or best_val_preds is None")
            logger.warning("   This may indicate an issue with validation during training")
        elif full_save_path is None:
            logger.warning("⚠️  Cannot save confusion matrix: full_save_path is None")
        elif len(best_val_labels) == 0 or len(best_val_preds) == 0:
            logger.warning("⚠️  Cannot generate confusion matrix: validation data is empty")
        else:
            # Calculate confusion matrix
            cm = confusion_matrix(best_val_labels, best_val_preds, labels=[0, 1, 2])
            cm_percentage = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10) * 100  # Add small epsilon to avoid division by zero
            
            # Log confusion matrix as text first
            logger.info("\nConfusion Matrix (Counts):")
            logger.info("=" * 50)
            header = "True\\Pred".ljust(15) + " | " + " | ".join([CLASS_NAMES.get(i, f"Class{i}").ljust(12) for i in range(len(CLASS_NAMES))])
            logger.info(header)
            logger.info("-" * 50)
            for i, class_name in CLASS_NAMES.items():
                row = class_name.ljust(15) + " | " + " | ".join([str(cm[i, j]).ljust(12) for j in range(len(CLASS_NAMES))])
                logger.info(row)
            
            logger.info("\nConfusion Matrix (Percentages):")
            logger.info("=" * 50)
            logger.info(header)
            logger.info("-" * 50)
            for i, class_name in CLASS_NAMES.items():
                row = class_name.ljust(15) + " | " + " | ".join([f"{cm_percentage[i, j]:.1f}%".ljust(12) for j in range(len(CLASS_NAMES))])
                logger.info(row)
            
            # Generate visualization
            plt.figure(figsize=(10, 8))
            plt.imshow(cm_percentage, interpolation='nearest', cmap='YlOrRd', vmin=0, vmax=100)
            plt.title(f'Confusion Matrix (Percentages) - F1: {best_val_f1:.4f}')
            plt.colorbar(label='Percentage')
            
            # Add text annotations with percentages
            thresh = 50  # Threshold for text color
            for i in range(cm_percentage.shape[0]):
                for j in range(cm_percentage.shape[1]):
                    plt.text(j, i, f'{cm_percentage[i, j]:.1f}%\n({int(cm[i, j])})',
                            ha="center", va="center",
                            color="white" if cm_percentage[i, j] > thresh else "black",
                            fontsize=9)
            
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.xticks(np.arange(len(CLASS_NAMES)), [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))], rotation=45)
            plt.yticks(np.arange(len(CLASS_NAMES)), [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))])
            plt.tight_layout()
            
            cm_save_path = os.path.join(
                os.path.dirname(full_save_path),
                f"{os.path.splitext(os.path.basename(full_save_path))[0]}__confusion_matrix.png",
            )
            plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Confusion matrix saved to {cm_save_path}")
            
            # Log per-class metrics
            logger.info("\nPer-Class Metrics:")
            for i, class_name in CLASS_NAMES.items():
                tn = sum((np.array(best_val_labels) != i) & (np.array(best_val_preds) != i))
                fp = sum((np.array(best_val_labels) != i) & (np.array(best_val_preds) == i))
                fn = sum((np.array(best_val_labels) == i) & (np.array(best_val_preds) != i))
                tp = sum((np.array(best_val_labels) == i) & (np.array(best_val_preds) == i))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                logger.info(f"\n{class_name.upper()}:")
                logger.info(f"Precision: {precision:.4f}")
                logger.info(f"Recall: {recall:.4f}")
                logger.info(f"F1-Score: {f1:.4f}")
                logger.info(f"True Positives: {tp}")
                logger.info(f"False Positives: {fp}")
                logger.info(f"False Negatives: {fn}")
                logger.info(f"True Negatives: {tn}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error during model retraining: {e}")
        return model
    finally:
        # Clean up GPU resources after retraining
        cleanup_gpu()
        stop_gpu_monitoring()

def process_single_file(zonal_file: str, model, feature_columns: list, config: dict, output_dirs: dict, clf=None, scaler=None) -> None:
    """Process a single zonal statistics file with predictions and rasterization."""
    try:
        original_name = zonal_file
        if isinstance(zonal_file, str) and zonal_file.startswith("s3://"):
            raise ValueError("S3 zonal statistics files are no longer supported. Please use local files.")
        local_path = zonal_file
        cleanup_needed = False

        base_name = os.path.basename(original_name).split('.')[0]
        
        # Check if prediction already exists for this file
        # In retraining mode (finetune_predict), we should overwrite existing predictions
        prediction_mode = config.get('prediction', {}).get('mode', 'load')
        overwrite_predictions = (prediction_mode == 'retrain' or 
                                config.get('prediction', {}).get('retraining', {}).get('enabled', False))
        
        raw_path = os.path.join(output_dirs['raw'], f"{base_name}_predicted.gpkg")
        prediction_exists = os.path.exists(raw_path)
        
        if prediction_exists:
            if overwrite_predictions:
                logger.info(f"Prediction already exists for {base_name}. Overwriting (retraining mode).")
            else:
                logger.info(f"Prediction already exists for {base_name}. Skipping.")
                return
        
        logger.info(f"Processing {base_name}")
        
        try:
            if local_path.lower().endswith(".parquet"):
                gdf = gpd.read_parquet(local_path)
            else:
                gdf = gpd.read_file(local_path)
        finally:
            if cleanup_needed:
                try:
                    os.remove(local_path)
                except OSError:
                    pass
        logger.info(f"Columns in {base_name}: {gdf.columns.tolist()}")
        
        if gdf is None or gdf.empty:
            return

        # Get input size from model metadata
        input_size = len(feature_columns)  # Use actual feature columns length
        if isinstance(model, dict):
            input_size = model.get('input_size', len(feature_columns))
        elif hasattr(model, 'input_size'):
            input_size = model.input_size
        
        # Build expected columns directly from provided list (align with training)
        expected_columns = feature_columns[:]
        
        # Identify categorical features - these should NOT be scaled
        # Categorical: UTM first_digit, hemisphere_code
        # Numerical spatial: utm_zone_number (NOT scaled, preserves spatial proximity)
        # 
        # H3 geo embeddings removed: hash-based H3 loses spatial proximity and creates
        # thousands of categorical values with no benefit. UTM zone_number already provides
        # spatial proximity.
        categorical_features = []
        if 'utm_first_digit' in expected_columns:
            categorical_features.append('utm_first_digit')
        if 'utm_hemisphere_code' in expected_columns:
            categorical_features.append('utm_hemisphere_code')
        # Backward compatibility: old columns
        if 'utm_hemisphere' in expected_columns:
            categorical_features.append('utm_hemisphere')
        if 'utm_zone' in expected_columns:
            categorical_features.append('utm_zone')
        
        # Numerical spatial features (preserve spatial proximity, NOT scaled)
        numerical_spatial_features = []
        if 'utm_zone_number' in expected_columns:
            numerical_spatial_features.append('utm_zone_number')
        if 'h3_cell_id_regional_numeric' in expected_columns:
            categorical_features.append('h3_cell_id_regional_numeric')
        if 'h3_cell_id_local_numeric' in expected_columns:
            categorical_features.append('h3_cell_id_local_numeric')
        if 'h3_cell_id_numeric' in expected_columns:
            categorical_features.append('h3_cell_id_numeric')
        
        continuous_features = [col for col in expected_columns 
                               if col not in categorical_features and col not in numerical_spatial_features]
        
        # H3 geo embeddings removed: hash-based H3 loses spatial proximity and creates
        # thousands of categorical values with no benefit. UTM zone_number already provides
        # spatial proximity. If H3 columns are present in data, they will be ignored.
        
        # Check if all expected columns are present
        missing_columns = [col for col in expected_columns if col not in gdf.columns]
        if missing_columns:
            logger.warning(f"Missing columns in {base_name}: {missing_columns}")
            # Fill missing columns with zeros
            for col in missing_columns:
                gdf[col] = 0
        
        # Prepare features using only the expected columns
        X = gdf[expected_columns].fillna(0)
        # Convert to float32 to reduce memory usage
        X = X.astype(np.float32)
        logger.info(f"Using feature columns: {expected_columns}")
        logger.info(f"Input shape: {X.shape}")
        
        # Filter out nodata values (65535) - ignore rows where any feature has nodata
        NODATA_VALUE = 65535
        nodata_mask = (X == NODATA_VALUE).any(axis=1)
        valid_mask = ~nodata_mask
        
        nodata_count = nodata_mask.sum()
        valid_count = valid_mask.sum()
        logger.info(f"NoData filtering: {valid_count:,} valid rows, {nodata_count:,} nodata rows (will be skipped)")
        
        # Initialize predictions and probabilities arrays for all rows
        predictions = np.full(len(X), -1, dtype=np.int32)  # Use -1 as nodata indicator
        probabilities = np.zeros((len(X), 3), dtype=np.float32)
        
        # Only process valid rows (skip nodata rows)
        if valid_count == 0:
            logger.warning(f"No valid rows found in {base_name} (all rows contain nodata). Skipping predictions.")
            # Set all to nodata
            gdf['predicted_class'] = -1
            gdf['confidence'] = 0.0
            gdf['class_name'] = None
        else:
            # Extract valid features for prediction - convert to numpy array immediately to release DataFrame reference
            X_valid = X[valid_mask].values.copy()  # Convert to numpy array to avoid DataFrame memory overhead
            del X  # Release the original DataFrame immediately
            scaler_transform_warned = False
            
            # Use the predictor's optimized batch size instead of config value
            # The predictor was already initialized with optimal batch size from profiling
            batch_size = clf.batch_size  # Use the predictor's batch_size (already optimized)
            memory_limit_mb = config.get('prediction', {}).get('memory_limit_mb', 20000)  # Default 20GB memory limit
            total_samples = len(X_valid)
            
            # Estimate memory usage and adjust batch size if needed
            estimated_memory_mb = (total_samples * len(expected_columns) * 4) / (1024 * 1024)  # Rough estimate in MB (float32 = 4 bytes)
            logger.info(f"Estimated memory usage: {estimated_memory_mb:.1f}MB for {total_samples:,} samples with {len(expected_columns)} features")
            
            if estimated_memory_mb > memory_limit_mb:
                # Adjust batch size to stay within memory limits
                # More conservative: account for input data, output probabilities, and intermediate arrays
                # Each sample: input (len(expected_columns) * 4 bytes) + output (3 * 4 bytes) + overhead
                bytes_per_sample = (len(expected_columns) * 4) + (3 * 4) + (len(expected_columns) * 4)  # input + output + intermediate
                optimal_batch_size = int((memory_limit_mb * 1024 * 1024) / (bytes_per_sample * 10))  # Factor of 10 for extra safety
                if optimal_batch_size < batch_size:
                    logger.warning(f"Estimated memory usage ({estimated_memory_mb:.1f}MB) exceeds limit ({memory_limit_mb}MB). Reducing batch size from {batch_size:,} to {optimal_batch_size:,}")
                    batch_size = max(500, optimal_batch_size)  # Reduced minimum batch size from 1000 to 500
                    # Update the predictor's batch size to match
                    clf.batch_size = batch_size
            
            if total_samples > batch_size:
                logger.info(f"Large dataset detected ({total_samples:,} valid samples). Using batch processing with batch_size={batch_size:,}")
                
                # Process in batches to avoid memory issues
                # Pre-allocate arrays to avoid memory fragmentation
                valid_probabilities = np.zeros((total_samples, 3), dtype=np.float32)
                valid_predictions = np.zeros(total_samples, dtype=np.int64)
                
                import gc
                
                for start_idx in range(0, total_samples, batch_size):
                    end_idx = min(start_idx + batch_size, total_samples)
                    # Extract batch directly from numpy array (X_valid is already a numpy array)
                    batch_X_values = X_valid[start_idx:end_idx].copy()
                    
                    # Apply scaler to batch if available (only continuous features, not categorical)
                    if scaler is not None:
                        try:
                            # Scale only continuous features; keep categorical features as-is
                            if continuous_features:
                                # Get indices of continuous features from expected_columns
                                continuous_indices = [expected_columns.index(col) for col in continuous_features if col in expected_columns]
                                if continuous_indices:
                                    batch_X_values[:, continuous_indices] = scaler.transform(batch_X_values[:, continuous_indices])
                        except Exception as e:
                            if not scaler_transform_warned:
                                logger.warning(f"Scaler transform failed; proceeding UN-SCALED for this file. Error: {e}")
                                scaler_transform_warned = True
                    
                    # Predict on batch - write directly to pre-allocated array
                    try:
                        # Pass the full output array and offset to write directly
                        batch_probs = clf.predict_proba(batch_X_values, output_array=valid_probabilities, output_offset=start_idx)
                        batch_preds = batch_probs.argmax(axis=1)
                        
                        # Write predictions directly
                        valid_predictions[start_idx:end_idx] = batch_preds.astype(np.int64)
                        
                        # Clean up batch data immediately - more aggressive cleanup
                        del batch_X_values, batch_probs, batch_preds
                        
                        # Log progress (less frequently to reduce I/O overhead)
                        progress = (end_idx / total_samples) * 100
                        if (start_idx // batch_size) % 10 == 0:  # Log every 10 batches (reduced from 5)
                            logger.info(f"Processed batch {start_idx//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size} ({progress:.1f}%)")
                        
                        # Force garbage collection every 5 batches to prevent memory accumulation (less frequent for speed)
                        if (start_idx // batch_size) % 5 == 0:  # Reduced from 3 to 5 for better performance
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                    except Exception as e:
                        logger.error(f"Batch prediction failed for indices {start_idx}-{end_idx}: {e}")
                        # Fallback: random predictions for this batch
                        batch_size_actual = end_idx - start_idx
                        batch_preds = np.random.randint(0, 3, size=batch_size_actual)
                        batch_probs = np.zeros((batch_size_actual, 3), dtype=np.float32)
                        for i, pred in enumerate(batch_preds):
                            batch_probs[i, pred] = 1.0
                        
                        valid_probabilities[start_idx:end_idx] = batch_probs
                        valid_predictions[start_idx:end_idx] = batch_preds
                        
                        del batch_X_values, batch_probs, batch_preds
                        gc.collect()
                        logger.warning(f"Using random predictions for batch {start_idx//batch_size + 1}")
                
                # Final cleanup
                gc.collect()
                
                logger.info(f"Batch processing completed. Final shapes: probs={valid_probabilities.shape}, preds={valid_predictions.shape}")
                
            else:
                # Small dataset - process all at once
                # X_valid is already a numpy array, apply scaler directly
                # Apply saved scaler if available (only continuous features, not categorical)
                try:
                    if scaler is not None:
                        # Scale only continuous features; keep categorical features as-is
                        if continuous_features:
                            # Get indices of continuous features from expected_columns
                            continuous_indices = [expected_columns.index(col) for col in continuous_features if col in expected_columns]
                            if continuous_indices:
                                X_valid[:, continuous_indices] = scaler.transform(X_valid[:, continuous_indices])
                except Exception as e:
                    if not scaler_transform_warned:
                        logger.warning(f"Scaler transform failed; proceeding UN-SCALED for this file. Error: {e}")
                        scaler_transform_warned = True
                
                # Predict via available classifier
                if clf is None:
                    raise RuntimeError("No classifier available for predictions.")
                
                try:
                    valid_probabilities = clf.predict_proba(X_valid)
                    valid_predictions = valid_probabilities.argmax(axis=1)
                    # Release X_valid after prediction
                    del X_valid
                    gc.collect()
                except Exception as e:
                    logger.error(f"Prediction failed: {e}")
                    # Fallback: assign random classes
                    np.random.seed(42)  # For reproducible fallback
                    valid_predictions = np.random.randint(0, 3, size=len(X_valid))
                    valid_probabilities = np.zeros((len(X_valid), 3))
                    for i, pred in enumerate(valid_predictions):
                        valid_probabilities[i, pred] = 1.0
                    del X_valid
                    gc.collect()
                    logger.warning("Using random class assignments as fallback")
            
            # Place valid predictions back into full arrays
            predictions[valid_mask] = valid_predictions
            probabilities[valid_mask] = valid_probabilities
            
            # Set nodata rows to -1 (nodata indicator) and confidence to 0
            predictions[nodata_mask] = -1
            probabilities[nodata_mask] = 0.0
        
        # Add predictions and confidence to dataframe
        gdf['predicted_class'] = predictions
        gdf['confidence'] = probabilities.max(axis=1)
        # Map class names, handling nodata (-1) case
        gdf['class_name'] = gdf['predicted_class'].apply(lambda x: CLASS_NAMES.get(x, None) if x >= 0 else None)
        
        # Clean up large arrays to free memory (X already deleted earlier when converted to X_valid)
        del probabilities, predictions
        gc.collect()
        
        # Save raw predictions (ensure overwrite by removing existing file first)
        try:
            if os.path.exists(raw_path):
                os.remove(raw_path)
        except Exception:
            pass

        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        gdf.to_file(raw_path, driver='GPKG')
        final_path = raw_path
        
        logger.info(f"Saved predictions to: {final_path}")
        
        # Process low confidence predictions - keep lowest N per class for balanced labeling
        # Exclude nodata polygons (predicted_class == -1) from confidence filtering
        valid_pred_mask = gdf['predicted_class'] >= 0
        valid_gdf = gdf[valid_pred_mask].copy()
        
        # Get per-class sample size from config (default: 1000 per class)
        retraining_config = config.get('prediction', {}).get('retraining', {})
        samples_per_class = retraining_config.get('samples_per_class', 1000)
        
        # Class mapping: 0=grassland, 1=tree, 2=urban
        class_ids = [1, 2, 0]  # tree, urban, grass(land) - order matches user's request
        class_names = ['tree', 'urban', 'grass']
        
        # Collect lowest confidence predictions per class
        # Goal: 1000 of each class (tree, urban, grass) = 3000 total features
        # If any class has < 1000, take ALL of that class to avoid errors
        selected_indices = []
        class_counts = {}
        
        for class_id, class_name in zip(class_ids, class_names):
            class_mask = valid_gdf['predicted_class'] == class_id
            class_predictions = valid_gdf[class_mask].copy()
            
            if len(class_predictions) == 0:
                logger.warning(f"No {class_name} predictions found - skipping this class")
                class_counts[class_name] = 0
                continue
            
            # Sort by confidence (ascending) to get LOWEST confidence first
            class_predictions_sorted = class_predictions.sort_values('confidence', ascending=True)
            
            # Take up to samples_per_class (1000), or ALL if fewer available
            n_available = len(class_predictions_sorted)
            n_samples = min(samples_per_class, n_available)
            selected_class = class_predictions_sorted.head(n_samples)
            
            selected_indices.extend(selected_class.index.tolist())
            class_counts[class_name] = n_samples
            
            if n_samples < samples_per_class:
                logger.info(
                    f"{class_name.capitalize()}: Selected ALL {n_samples} predictions "
                    f"(fewer than target {samples_per_class}, confidence range: "
                    f"{selected_class['confidence'].min():.3f} - {selected_class['confidence'].max():.3f})"
                )
            else:
                logger.info(
                    f"{class_name.capitalize()}: Selected {n_samples} lowest confidence predictions "
                    f"(out of {n_available} total, confidence range: "
                    f"{selected_class['confidence'].min():.3f} - {selected_class['confidence'].max():.3f})"
                )
        
        # Create mask for selected predictions
        low_conf_mask = gdf.index.isin(selected_indices)
        low_conf_pixels = low_conf_mask.sum()
        total_pixels = len(gdf)
        valid_pixels = valid_pred_mask.sum()
        nodata_pixels = (~valid_pred_mask).sum()
        
        # Validate expected output: should be ~3000 (1000 per class) or less if some classes are missing
        expected_total = sum(class_counts.values())
        
        logger.info(f"Confidence selection statistics for {base_name}:")
        logger.info(f"Total pixels: {total_pixels} (valid: {valid_pixels}, nodata: {nodata_pixels})")
        logger.info(f"Selected for labeling: {low_conf_pixels} features (expected: {expected_total})")
        for class_name, count in class_counts.items():
            logger.info(f"  - {class_name.capitalize()}: {count} samples")
        
        # Verify we got the expected count
        if low_conf_pixels != expected_total:
            logger.warning(
                f"Mismatch: Selected {low_conf_pixels} features but expected {expected_total}. "
                f"This may indicate duplicate indices or filtering issues."
            )
        
        # Only save selected low confidence polygons that are not nodata
        if low_conf_mask.any():
            # Filter to only selected polygons (already excludes nodata)
            low_conf_gdf = gdf[low_conf_mask].copy()
            
            # Create descriptive filename with timestamp (avoid very long class-count suffix)
            # Old pattern: *_low_confidence_tree1000_urban1000_grass1000.gpkg
            # New pattern: *_low_confidence_<YYYYMMDDTHHMMSS>.gpkg
            ts = datetime.now().strftime("%Y%m%dT%H%M%S")
            confidence_filename = f"{base_name}_low_confidence_{ts}.gpkg"
            
            # Ensure class_id column exists for retraining
            if 'class_id' not in low_conf_gdf.columns:
                low_conf_gdf['class_id'] = None
            elif 'class_id' in gdf.columns:
                low_conf_gdf['class_id'] = gdf.loc[low_conf_gdf.index, 'class_id']
            
            confidence_path = os.path.join(output_dirs['confidence'], confidence_filename)
            
            try:
                if os.path.exists(confidence_path):
                    os.remove(confidence_path)
            except Exception:
                pass

            os.makedirs(os.path.dirname(confidence_path), exist_ok=True)
            low_conf_gdf.to_file(confidence_path, driver='GPKG')
            
            # Verify output file has expected feature count
            actual_feature_count = len(low_conf_gdf)
            expected_feature_count = sum(class_counts.values())
            if actual_feature_count != expected_feature_count:
                logger.warning(
                    f"Feature count mismatch in output file: {actual_feature_count} features written, "
                    f"expected {expected_feature_count}. This may indicate data filtering issues."
                )
            else:
                logger.info(
                    f"✅ Confidence file created with {actual_feature_count} features "
                    f"({class_counts.get('tree', 0)} tree, {class_counts.get('urban', 0)} urban, "
                    f"{class_counts.get('grass', 0)} grass)"
                )
            
            final_confidence_path = confidence_path
            
            logger.info(f"Saved {len(low_conf_gdf)} low confidence polygons to {final_confidence_path}")
            
            # Return low confidence data for potential retraining
            confidence_gdf = low_conf_gdf
        else:
            logger.info(f"No low confidence polygons found in {base_name} - no confidence file created")
            confidence_gdf = None
        
        # Generate raster outputs (optional - can be disabled via config)
        generate_rasters = config.get('prediction', {}).get('generate_rasters', True)
        if not generate_rasters:
            logger.info("Raster generation disabled in config. Only GPKG outputs will be created.")
            return confidence_gdf  # Return low confidence data or None
            
        try:
            # Get the bounds of the data
            bounds = gdf.total_bounds
            resolution = 3  # 3m resolution
            width = int((bounds[2] - bounds[0]) / resolution)
            height = int((bounds[3] - bounds[1]) / resolution)
            transform = from_origin(bounds[0], bounds[3], resolution, resolution)

            # Create multi-class raster
            # Filter out nodata polygons (predicted_class == -1) before rasterization
            valid_gdf = gdf[gdf['predicted_class'] >= 0].copy()
            if len(valid_gdf) == 0:
                logger.warning(f"No valid predictions to rasterize for {base_name} (all are nodata)")
            else:
                logger.info(f"Rasterizing {len(valid_gdf)} valid polygons (excluding {len(gdf) - len(valid_gdf)} nodata polygons)")
                shapes = ((geom, int(value)) for geom, value in zip(valid_gdf.geometry, valid_gdf['predicted_class']))
                raster = rasterio.features.rasterize(
                    shapes=shapes,
                    out_shape=(height, width),
                    transform=transform,
                    fill=255,  # Use 255 as nodata fill value for uint8
                    dtype=np.uint8
                )

                raster_path = os.path.join(output_dirs['raster']['multiclass'], f"{base_name}_classification.tif")
                with rasterio.open(
                    raster_path,
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=1,
                    dtype=np.uint8,
                    crs=gdf.crs,
                    transform=transform,
                    compress='deflate',
                    zlevel=9,
                    tiled=True,
                    blockxsize=256,
                    blockysize=256,
                    nodata=255  # Set nodata value to 255
                ) as dst:
                    dst.write(raster, 1)
                    # Add class colormap:
                    # - tree: red, 30% opacity
                    # - grassland: yellow, 5% opacity
                    # - urban: blue, 5% opacity
                    dst.write_colormap(1, {
                        0: (255, 255, 0, 13),   # grassland - yellow, ~5% opacity
                        1: (255,   0, 0, 77),   # tree - red, ~30% opacity
                        2: (  0,   0, 255, 13), # urban - blue, ~5% opacity
                    })
                logger.info(f"Saved multi-class raster to {raster_path}")

                # Create binary raster (tree vs non-tree)
                # Use same valid_gdf (already filtered for nodata)
                logger.info(f"Creating binary classification raster at {resolution}m resolution...")
                binary_shapes = ((geom, 1 if value == 'tree' else 0) for geom, value in zip(valid_gdf.geometry, valid_gdf['class_name']))
                binary_raster = rasterio.features.rasterize(
                    shapes=binary_shapes,
                    out_shape=(height, width),
                    transform=transform,
                    fill=255,  # Use 255 as nodata fill value for uint8
                    dtype=np.uint8
                )

                binary_raster_path = os.path.join(output_dirs['raster']['binary'], f"{base_name}_binary.tif")
                with rasterio.open(
                    binary_raster_path,
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=1,
                    dtype=np.uint8,
                    crs=gdf.crs,
                    transform=transform,
                    compress='deflate',
                    zlevel=9,
                    tiled=True,
                    blockxsize=256,
                    blockysize=256,
                    nodata=255  # Set nodata value to 255
                ) as dst:
                    dst.write(binary_raster, 1)
                    # Add binary class colormap:
                    # - tree pixels in red (~30% opacity)
                    # - background very faint (~5% opacity)
                    dst.write_colormap(1, {
                        0: (0,   0,   0, 13),   # no_tree - nearly transparent
                        1: (255, 0,   0, 77),   # tree - red, ~30% opacity
                    })
                logger.info(f"Saved binary raster to {binary_raster_path}")

        except Exception as e:
            logger.warning(f"Error creating raster outputs: {e}. Continuing with GPKG outputs only.")
            # Don't raise - allow pipeline to continue with GPKG outputs

        return confidence_gdf if low_conf_mask.any() else None
        
    except Exception as e:
        logger.error(f"Error processing {zonal_file}: {str(e)}")
        return None

def make_predictions(
    model,
    zonal_stats_dir: str,
    predictions_dir: str,
    config: dict,
    retraining_confidence_dir: str | None = None,
) -> None:
    """Make predictions using the provided model.

    When retraining_confidence_dir is set (e.g. in time_series mode), labeled files for
    retraining are read from that directory instead of predictions_dir/confidence. Use
    peak_vitality/predictions/confidence in time_series since only peak_vitality runs
    prediction during the confidence step.
    """
    if isinstance(predictions_dir, str) and predictions_dir.startswith("s3://"):
        raise ValueError("S3 prediction output directories are no longer supported. Please use a local directory.")

    output_dirs = {
        'raw': os.path.join(predictions_dir, "raw"),
        'confidence': os.path.join(predictions_dir, "confidence"),
        'polygons': os.path.join(predictions_dir, "polygons"),
        'raster': {
            'multiclass': os.path.join(predictions_dir, "raster", "multiclass"),
            'binary': os.path.join(predictions_dir, "raster", "binary")
        }
    }
    for directory in [output_dirs['raw'], output_dirs['confidence'], output_dirs['polygons'],
                     output_dirs['raster']['multiclass'], output_dirs['raster']['binary']]:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Force PyTorch-based prediction path (remove GBDT usage)
    logger.info("Using PyTorch-based predictor for tabular predictions (GBDT disabled)")
    
    # Clear GPU memory before starting prediction to ensure clean state
    if torch.cuda.is_available():
        cleanup_gpu()
        logger.info("GPU memory cleared before prediction setup")
    
    class PyTorchPredictor:
        def __init__(self, pytorch_model, feature_columns, batch_size: int = 2000, gpu_max_rows: int = 500000):
            self.pytorch_model = pytorch_model
            self.feature_columns = feature_columns
            # Force GPU usage - raise error if CUDA not available
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. GPU is required for prediction.")
            self.device = torch.device('cuda')
            # Tunables - reduced batch_size to prevent memory accumulation (from 8000 to 2000)
            self.batch_size = max(1, int(batch_size))
            self.gpu_max_rows = max(1, int(gpu_max_rows))
            
            # Clear GPU memory before moving model to ensure clean state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Ensure model is on the correct device and in eval mode
            self.pytorch_model = self.pytorch_model.to(self.device).eval()
            
            # Log device being used and GPU memory status
            if self.device.type == 'cuda':
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                allocated_mem = torch.cuda.memory_allocated() / (1024**3)
                free_mem = total_mem - allocated_mem
                logger.info(f"PyTorchPredictor initialized on GPU (Total: {total_mem:.1f}GB, Allocated: {allocated_mem:.1f}GB, Free: {free_mem:.1f}GB)")
            else:
                logger.info("PyTorchPredictor initialized on CPU")
            
        def _move_model_to_device(self, device):
            """Safely move model to specified device and ensure it's in eval mode."""
            if self.pytorch_model is not None:
                current_device = next(self.pytorch_model.parameters()).device
                # Only allow GPU device - no CPU fallback
                if device.type != 'cuda':
                    raise RuntimeError(f"GPU device required, but got {device.type}. CUDA must be available.")
                
                if current_device.type != 'cuda':
                    logger.info(f"Moving model to GPU")
                else:
                    logger.debug(f"Model already on GPU")
                
                self.pytorch_model = self.pytorch_model.to(device).eval()
                torch.cuda.synchronize()
            
        def _predict_batches(self, X: np.ndarray, device, output_array: np.ndarray | None = None, output_offset: int = 0) -> np.ndarray:
            """
            Predict in batches, optionally writing directly to pre-allocated output array.
            
            Args:
                X: Input features array
                device: Torch device to use
                output_array: Optional pre-allocated array to write results to
                output_offset: Offset in output_array where to start writing (default: 0)
            
            Returns:
                Probabilities array (either output_array slice if provided, or newly allocated)
            """
            num_samples = X.shape[0]
            num_classes = self.pytorch_model.final[-1].out_features if hasattr(self.pytorch_model, 'final') else 3
            
            # Pre-allocate output if not provided
            if output_array is None:
                output_array = np.zeros((num_samples, num_classes), dtype=np.float32)
                output_offset = 0
            elif output_array.shape[0] < output_offset + num_samples:
                raise ValueError(f"output_array too small: shape {output_array.shape}, need at least {output_offset + num_samples} rows")
            
            import gc
            
            with torch.no_grad():
                # Ensure model is on the correct device before prediction
                if next(self.pytorch_model.parameters()).device != device:
                    self._move_model_to_device(device)
                
                for start_idx in range(0, X.shape[0], self.batch_size):
                    end_idx = min(start_idx + self.batch_size, X.shape[0])
                    
                    # Clear cache before each batch to prevent accumulation
                    if torch.cuda.is_available() and device.type == 'cuda':
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Create tensor and move to device with non_blocking for efficiency
                    X_tensor = torch.from_numpy(X[start_idx:end_idx]).float()
                    if device.type == 'cuda':
                        X_tensor = X_tensor.to(device, non_blocking=True)
                    
                    # Forward pass - ensure model is in eval mode and use no_grad
                    # Use inference_mode() for even better memory efficiency (disables autograd completely)
                    self.pytorch_model.eval()
                    with torch.inference_mode():  # More memory efficient than no_grad()
                        logits = self.pytorch_model(X_tensor)
                        # Move logits to CPU immediately to free GPU memory before softmax
                        logits_cpu = logits.cpu()
                        del logits  # Free GPU memory immediately
                        probs = torch.softmax(logits_cpu, dim=1).numpy().astype(np.float32)
                        del logits_cpu  # Free CPU memory
                    
                    # Write directly to output array at correct offset
                    output_array[output_offset + start_idx:output_offset + end_idx] = probs
                    
                    # Free per-batch tensors ASAP - more aggressive cleanup
                    del X_tensor, probs
                    if torch.cuda.is_available() and device.type == 'cuda':
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Aggressive memory cleanup every batch
                    gc.collect()
            
            # Return the slice that was written to
            if output_offset == 0 and output_array.shape[0] == num_samples:
                return output_array
            else:
                return output_array[output_offset:output_offset + num_samples]
            
        def predict_proba(self, X, output_array: np.ndarray | None = None, output_offset: int = 0):
            """Batched prediction with automatic CPU fallback for very large inputs or OOM.
            
            Args:
                X: Input features array
                output_array: Optional pre-allocated array to write results to
                output_offset: Offset in output_array where to start writing (default: 0)
            
            Returns:
                Probabilities array (either output_array slice if provided, or newly allocated)
            """
            try:
                # Force GPU usage - no CPU fallback
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available. GPU is required for prediction.")
                
                # Ensure we're using GPU
                if self.device.type != 'cuda':
                    self.device = torch.device('cuda')
                    self._move_model_to_device(self.device)
                
                # Clear GPU cache first to ensure we have maximum available memory
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Estimate memory needed: 
                # - Model weights (rough estimate: ~50MB for DeepUNetClassifier)
                # - Input batch: batch_size * features * 4 bytes
                # - Output logits: batch_size * num_classes * 4 bytes  
                # - Intermediate activations: ~2x input size
                # - Overhead: ~20% buffer
                model_memory_mb = 50  # Rough estimate for model
                input_memory_mb = (self.batch_size * len(self.feature_columns) * 4) / (1024 * 1024)
                output_memory_mb = (self.batch_size * 3 * 4) / (1024 * 1024)  # 3 classes
                intermediate_memory_mb = input_memory_mb * 2  # Intermediate activations
                estimated_memory_mb = (model_memory_mb + input_memory_mb + output_memory_mb + intermediate_memory_mb) * 1.2  # 20% overhead
                
                # Get actual free GPU memory from nvidia-smi (accounts for other processes)
                total_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                allocated_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                
                # Try to get actual free memory from nvidia-smi (only log once per call, not per batch)
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True, timeout=2)
                    if result.returncode == 0:
                        actual_free_mb = float(result.stdout.strip())
                        free_memory_mb = actual_free_mb
                    else:
                        # Fallback to torch calculation
                        free_memory_mb = total_memory_mb - allocated_memory_mb
                except Exception:
                    # Fallback to torch calculation
                    free_memory_mb = total_memory_mb - allocated_memory_mb
                
                # Only log memory status once per predict_proba call (not for every batch)
                # Use debug level to reduce verbosity during normal operation
                logger.debug(f"GPU memory: Total={total_memory_mb:.1f}MB, Allocated={allocated_memory_mb:.1f}MB, Free={free_memory_mb:.1f}MB, Estimated needed={estimated_memory_mb:.1f}MB")
                
                # Check if we have enough memory, but still use GPU (reduce batch size if needed)
                if free_memory_mb < 1024:
                    raise RuntimeError(f"GPU memory too low ({free_memory_mb:.1f}MB free < 1GB). Cannot proceed.")
                
                # If estimated memory exceeds available, reduce batch size dynamically
                if estimated_memory_mb > free_memory_mb * 0.8:
                    # Reduce batch size to fit in available memory
                    safe_batch_size = int((free_memory_mb * 0.8 * 1024 * 1024) / (len(self.feature_columns) * 4 * 3))
                    safe_batch_size = max(100, safe_batch_size)  # Minimum batch size of 100
                    if safe_batch_size < self.batch_size:
                        logger.warning(f"Reducing batch size from {self.batch_size} to {safe_batch_size} to fit in GPU memory")
                        original_batch_size = self.batch_size
                        self.batch_size = safe_batch_size
                        result = self._predict_batches(X, self.device, output_array, output_offset)
                        self.batch_size = original_batch_size  # Restore original
                        return result
                
                # Only log once per predict_proba call (use debug to reduce verbosity)
                logger.debug(f"Using GPU for prediction (estimated {estimated_memory_mb:.1f}MB needed, {free_memory_mb:.1f}MB free)")
                
                # Ensure model is on GPU
                if next(self.pytorch_model.parameters()).device != self.device:
                    self._move_model_to_device(self.device)
                
                return self._predict_batches(X, self.device, output_array, output_offset)
            except RuntimeError as e:
                # CUDA OOM - try to recover by clearing cache and retrying with smaller batch
                error_str = str(e)
                if 'CUDA out of memory' in error_str or 'CUDA' in error_str:
                    logger.warning(f"GPU OOM error encountered. Clearing cache and retrying with reduced batch size.")
                    # Aggressive cleanup
                    cleanup_gpu()
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # Check actual free memory after cleanup using nvidia-smi
                    free_mem = None
                    if torch.cuda.is_available():
                        try:
                            import subprocess
                            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], 
                                                  capture_output=True, text=True, timeout=2)
                            if result.returncode == 0:
                                free_mem = float(result.stdout.strip())
                                logger.info(f"After cleanup: Actual free GPU memory = {free_mem:.1f}MB")
                        except Exception:
                            pass
                        
                        if free_mem is None:
                            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                            allocated_mem = torch.cuda.memory_allocated() / (1024**2)
                            free_mem = total_mem - allocated_mem
                            logger.info(f"After cleanup: Free GPU memory = {free_mem:.1f}MB")
                        
                        # Calculate safe batch size based on actual free memory
                        # Based on profiling: batch_size=100 needs ~1.1GB, batch_size=500 needs ~5.4GB
                        # Memory scales roughly linearly: ~10.7GB per 1000 samples
                        # Use very conservative estimate: reserve 1GB for model/overhead, use only 20% of remaining
                        usable_mem = max(0, (free_mem - 1024) * 0.2)  # Very conservative: only 20% of free memory
                        # Estimate: ~10.7MB per sample (from profiling: 10.7GB for 1000 samples)
                        safe_batch_size = max(10, int((usable_mem * 1024 * 1024) / (10.7 * 1024 * 1024)))
                        # Cap at 100 to be safe (we know 100 works from profiling)
                        safe_batch_size = min(safe_batch_size, 100)
                        logger.info(f"Calculated safe batch size: {safe_batch_size} (from {free_mem:.1f}MB free, {usable_mem:.1f}MB usable)")
                    else:
                        safe_batch_size = max(10, self.batch_size // 8)  # Reduce to 1/8 if can't check memory
                    
                    # Reduce batch size significantly and retry
                    original_batch_size = self.batch_size
                    self.batch_size = safe_batch_size
                    logger.warning(f"Retrying with reduced batch size: {self.batch_size} (original: {original_batch_size})")
                    try:
                        result = self._predict_batches(X, self.device, output_array, output_offset)
                        self.batch_size = original_batch_size  # Restore original
                        return result
                    except RuntimeError as retry_error:
                        self.batch_size = original_batch_size
                        # If still OOM, try even smaller batch (50 or 25)
                        if 'CUDA out of memory' in str(retry_error):
                            logger.error(f"Still OOM with batch size {safe_batch_size}. Trying even smaller batch size: 50")
                            self.batch_size = 50
                            try:
                                result = self._predict_batches(X, self.device, output_array, output_offset)
                                self.batch_size = original_batch_size
                                return result
                            except RuntimeError:
                                logger.error(f"Still OOM with batch size 50. Trying batch size 25")
                                self.batch_size = 25
                                try:
                                    result = self._predict_batches(X, self.device, output_array, output_offset)
                                    self.batch_size = original_batch_size
                                    return result
                                except RuntimeError as final_error:
                                    self.batch_size = original_batch_size
                                    logger.error(f"GPU OOM even with batch size 25. Process may have too much memory allocated. Check with: nvidia-smi")
                                    raise RuntimeError(f"GPU OOM even after reducing batch size to 25. Error: {final_error}")
                        raise RuntimeError(f"GPU OOM even after reducing batch size to {safe_batch_size}. Error: {retry_error}")
                raise
            except Exception as e:
                import traceback
                error_traceback = traceback.format_exc()
                logger.error(f"Error in PyTorch prediction: {e}")
                logger.error(f"Full traceback:\n{error_traceback}")
                # Don't return fallback - re-raise to see the actual error
                raise RuntimeError(f"PyTorch prediction failed: {e}") from e
    
    # Defer creating the PyTorch predictor until feature_columns are resolved below
    clf = None
    
    # ------------------------------------------------------------------
    # Resolve feature_columns to match TRAINING EXACTLY.
    # Priority:
    #   1) checkpoint['feature_columns']  (authoritative source)
    #   2) config.prediction.feature_columns (if explicitly overridden)
    #   3) default band means b02_mean..b11_mean (last resort)
    # ------------------------------------------------------------------
    feature_columns: list[str] = []
    checkpoint_data = None
    
    # 1) Try to read feature_columns from checkpoint metadata
    try:
        ckpt_path = config.get('prediction', {}).get('model_load_path')
        if ckpt_path:
            actual_ckpt_path = ckpt_path
            if isinstance(ckpt_path, str) and ckpt_path.startswith("s3://"):
                raise ValueError("S3 model checkpoints are no longer supported. Please use a local checkpoint path.")
            if not os.path.exists(ckpt_path):
                actual_ckpt_path = None
            
            if actual_ckpt_path and os.path.exists(actual_ckpt_path):
                checkpoint_data = _torch_load(actual_ckpt_path, map_location=torch.device('cpu'))
                checkpoint_feature_columns = checkpoint_data.get('feature_columns')
                if isinstance(checkpoint_feature_columns, list) and checkpoint_feature_columns:
                    feature_columns = checkpoint_feature_columns
                    logger.info(f"Using feature columns from checkpoint (authoritative): {feature_columns}")
    except Exception as e:
        logger.warning(f"Could not load feature columns from checkpoint: {e}")
    
    # 2) If checkpoint didn't provide them, fall back to config.
    if not feature_columns:
        cfg_cols = config.get('prediction', {}).get('feature_columns', [])
        cfg_cols = [c for c in cfg_cols if c != 'count']
        if cfg_cols:
            feature_columns = cfg_cols
            logger.info(f"Using feature columns from config (no checkpoint feature_columns present): {feature_columns}")
    
    # 3) Final fallback: default band means if nothing was found
    if not feature_columns:
        logger.warning("No feature columns found in checkpoint or config. Falling back to default band means b02_mean..b11_mean.")
        feature_columns = [f"b{idx:02d}_mean" for idx in range(2, 12)]
    logger.info(f"Using features: {feature_columns}")
    logger.info("Note: Predictions will be saved as GPKG files for visualization. Raster generation is optional.")
    
    # Create the PyTorch-based predictor now that feature_columns are known
    global _global_scaler
    scaler = None  # Ensure scaler is defined for downstream closures
    try:
        ckpt_path = config.get('prediction', {}).get('model_load_path')
        if not ckpt_path:
            raise FileNotFoundError(f"Model checkpoint path not specified in config")
        
        actual_ckpt_path = ckpt_path
        if isinstance(ckpt_path, str) and ckpt_path.startswith("s3://"):
            raise ValueError("S3 model checkpoints are no longer supported. Please use a local checkpoint path.")
        elif not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")
        
        # Force GPU usage - raise error if CUDA not available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. GPU is required for model loading and prediction.")
        device = torch.device('cuda')
        
        # Use already loaded checkpoint data if available, otherwise load it
        if checkpoint_data is not None:
            checkpoint = checkpoint_data
        else:
            checkpoint = _torch_load(actual_ckpt_path, map_location=device)
        
        # Use the same architecture as training: UNetClassifier with correct base_channels
        input_size = len(feature_columns)
        # Prefer checkpoint-provided num_classes if present and valid
        ckpt_num_classes = checkpoint.get('num_classes') if isinstance(checkpoint, dict) else None
        inferred_num_classes = None
        try:
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', {})
                # DeepUNetClassifier uses final.8 (last layer in Sequential)
                if 'final.8.weight' in state_dict:
                    inferred_num_classes = state_dict['final.8.weight'].shape[0]
                    logger.info(f"Inferred num_classes={inferred_num_classes} from final.8.weight (DeepUNetClassifier)")
                # UNetClassifier uses final.4 (last layer in Sequential)
                elif 'final.4.weight' in state_dict:
                    inferred_num_classes = state_dict['final.4.weight'].shape[0]
                    logger.info(f"Inferred num_classes={inferred_num_classes} from final.4.weight (UNetClassifier)")
        except Exception as e:
            logger.warning(f"Failed to infer num_classes from checkpoint weights: {e}")
        num_classes = inferred_num_classes or ckpt_num_classes or 3  # grassland, tree, urban
        logger.info(f"Using num_classes={num_classes} (inferred={inferred_num_classes}, checkpoint={ckpt_num_classes})")
        
        # Infer base_channels from checkpoint weights to match exactly
        inferred_base = None
        try:
            if 'encoder.input_proj.0.weight' in checkpoint['model_state_dict']:
                out_features = checkpoint['model_state_dict']['encoder.input_proj.0.weight'].shape[0]
                if out_features % 2 == 0:
                    inferred_base = out_features // 2
                    logger.info(f"Inferred base_channels={inferred_base} from checkpoint weights")
        except Exception:
            pass
        
        if inferred_base is not None and inferred_base > 0:
            torch_model = DeepUNetClassifier(input_size=input_size, num_classes=num_classes, base_channels=inferred_base)
            logger.info(f"Created DeepUNetClassifier with base_channels={inferred_base}")
        else:
            # Fallback to UNetClassifier (which uses base_channels=33)
            torch_model = UNetClassifier(input_size=input_size, num_classes=num_classes)
            logger.info(f"Created UNetClassifier (default base_channels=33)")
        
        logger.info(f"Model architecture: input_size={input_size}, num_classes={num_classes}")
        logger.info(f"Feature columns: {feature_columns}")
        state_dict = checkpoint.get('model_state_dict')
        
        if state_dict:
            try:
                torch_model.load_state_dict(state_dict)
                logger.info("Successfully loaded model state dict")
            except Exception as load_err:
                logger.warning(f"Strict load_state_dict failed: {load_err}")
                # Fallback 1: load only matching keys with matching shapes
                try:
                    model_state = torch_model.state_dict()
                    matched = 0
                    for k, v in state_dict.items():
                        if k in model_state and model_state[k].shape == v.shape:
                            model_state[k] = v
                            matched += 1
                    torch_model.load_state_dict(model_state, strict=False)
                    logger.info(f"Loaded {matched} matching tensors from checkpoint (strict=False)")
                except Exception as partial_err:
                    logger.warning(f"Partial load failed: {partial_err}. Proceeding with randomly initialized layers for mismatched shapes.")
        
        # Build scaler from checkpoint if available
        logger.info("🔍 Starting scaler reconstruction...")
        scaler_mean = checkpoint.get('scaler_mean')
        scaler_scale = checkpoint.get('scaler_scale')
        logger.info(f"=== SCALER RECONSTRUCTION DEBUG ===")
        logger.info(f"Checkpoint scaler_mean present: {scaler_mean is not None}")
        logger.info(f"Checkpoint scaler_scale present: {scaler_scale is not None}")
        logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        if scaler_mean is not None and scaler_scale is not None:
            logger.info("🔍 Attempting scaler reconstruction...")
            try:
                _global_scaler = StandardScaler()
                _global_scaler.mean_ = np.array(scaler_mean)
                _global_scaler.scale_ = np.array(scaler_scale)
                _global_scaler.var_ = _global_scaler.scale_ ** 2
                # Mark scaler as "fitted" for sklearn checks (required for transform() in newer sklearn)
                # Without this, scaler.transform(...) may raise NotFittedError and inference will run unscaled.
                _global_scaler.n_features_in_ = int(_global_scaler.mean_.shape[0])
                scaler = _global_scaler  # Local reference
                logger.info("✅ Successfully reconstructed StandardScaler from checkpoint stats")
                logger.info(f"Scaler mean shape: {scaler.mean_.shape}")
                logger.info(f"Scaler scale shape: {scaler.scale_.shape}")
            except Exception as e:
                logger.error(f"❌ Failed to reconstruct scaler: {e}")
                scaler = None
        else:
            logger.warning("❌ No scaler stats found in checkpoint - retraining will use unscaled features")
            scaler = None
        
        logger.info(f"Final scaler value: {scaler}")
        logger.info(f"=== END SCALER RECONSTRUCTION DEBUG ===")
        
        # Attach feature columns and scaler for downstream retraining logic
        torch_model.feature_columns = feature_columns
        torch_model.scaler = scaler  # Attach scaler for retraining (now properly reconstructed)
        logger.info(f"Attached scaler to torch_model: {torch_model.scaler}")
        torch_model = torch_model.to(device).eval()
        
        # Phase 1 Optimization: Compile model for faster inference (PyTorch 2.0+)
        # Note: Compilation is disabled by default due to potential compatibility issues
        # Enable by setting prediction.enable_model_compilation=true in config
        enable_compilation = config.get('prediction', {}).get('enable_model_compilation', False)
        if enable_compilation:
            try:
                if hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
                    logger.info("Compiling model with torch.compile() for faster inference...")
                    torch_model = torch.compile(torch_model, mode='reduce-overhead')
                    logger.info("Model compilation successful - expect 1.5-2x speedup in inference")
                else:
                    logger.debug("torch.compile() not available (requires PyTorch 2.0+). Skipping compilation.")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}. Continuing without compilation.")
        else:
            logger.debug("Model compilation disabled (set prediction.enable_model_compilation=true to enable)")
        
        # Configure batching and CPU fallback thresholds from config
        pred_cfg = config.get('prediction', {})
        # Phase 1 Optimization: Use optimal batch size with profiling
        # Note: Profiling can be disabled by setting prediction.disable_batch_profiling=true
        default_batch_size = int(pred_cfg.get('batch_size', 200))
        disable_profiling = pred_cfg.get('disable_batch_profiling', False)
        if disable_profiling:
            batch_size = default_batch_size
            logger.info(f"Batch size profiling disabled. Using config batch_size={batch_size}")
        else:
            try:
                batch_size = _find_optimal_batch_size(torch_model, feature_columns, default_batch_size, device)
            except Exception as e:
                logger.warning(f"Batch size profiling failed: {e}. Using default batch_size={default_batch_size}")
                batch_size = default_batch_size
        gpu_max_rows = int(pred_cfg.get('gpu_max_rows', 1000000))  # Increased threshold to allow GPU usage for larger datasets
        
        # Clear GPU memory before initializing predictor to ensure clean state
        if torch.cuda.is_available():
            cleanup_gpu()
            logger.info("GPU memory cleared before prediction initialization")
        
        clf = PyTorchPredictor(torch_model, feature_columns, batch_size=batch_size, gpu_max_rows=gpu_max_rows)
        logger.info(f"Initialized PyTorch predictor for predictions with optimized batch_size={batch_size}")
    except Exception as e:
        logger.error(f"Failed to initialize PyTorch predictor: {e}")
        raise

    # First, check if we have labeled data in confidence directory (enable retraining)
    # In time_series mode, use retraining_confidence_dir (peak_vitality/predictions/confidence)
    # so retraining finds labels even when predict() is called for current.
    confidence_dir = retraining_confidence_dir if retraining_confidence_dir else output_dirs["confidence"]
    if isinstance(confidence_dir, str) and confidence_dir.startswith("s3://"):
        raise ValueError("S3 confidence directories are no longer supported. Please use a local directory.")
    glob_pattern = os.path.join(confidence_dir, "*.gpkg")
    labeled_files: list[str] = glob.glob(glob_pattern)
    # Check retraining configuration with detailed logging
    pred_config = config.get('prediction', {})
    retraining_config = pred_config.get('retraining', {})
    mode = pred_config.get('mode', '')
    retraining_enabled = retraining_config.get('enabled', False)
    
    retrain_enabled = (
        mode in ['retrain', 'retrain_only', 'retrain_and_predict']
        or retraining_enabled
    )
    
    # Verbose logging on what we found and the conditions for retraining
    logger.info(f"Retraining config check:")
    logger.info(f"  prediction.mode: {mode}")
    logger.info(f"  prediction.retraining.enabled: {retraining_enabled}")
    logger.info(f"  Retraining enabled: {retrain_enabled}")
    logger.info(f"Confidence directory: {confidence_dir}")
    logger.info(f"Glob pattern for labeled files: {glob_pattern}")
    logger.info(f"Found {len(labeled_files)} candidate confidence files")
    for fp in labeled_files:
        logger.info(f"- Candidate file: {fp}")

    if not retrain_enabled:
        logger.info("Retraining disabled by config. Skipping labeled data scan.")

    # Track whether retraining actually happened in this call
    retraining_occurred = False
    
    # Check if retraining has already occurred in this workflow run (module-level flag)
    global _retraining_already_occurred
    if _retraining_already_occurred:
        logger.info("Retraining has already occurred in this workflow run. Skipping retraining to avoid duplicate training.")
        retrain_enabled = False
    
    if labeled_files and retrain_enabled:
        try:
            logger.info("Retraining enabled and labeled confidence files detected. Preparing data...")
            
            # Convert labeled confidence GPKG files to Parquet for efficient retraining
            from treelance_sentinel.imagery_processing import convert_zonal_stats_gpkg_to_parquet
            
            labeled_gdfs = []
            for fp in labeled_files:
                try:
                    # Check if file has labels before converting
                    try:
                        # Quick check: try to read only labeled rows
                        test_gdf = gpd.read_file(fp, where="class_id IS NOT NULL")
                        if test_gdf.empty or 'class_id' not in test_gdf.columns:
                            logger.info(f"Skipping {fp}: no labeled data found")
                            continue
                    except Exception:
                        # Fallback: read full file to check
                        test_gdf_full = gpd.read_file(fp)
                        if 'class_id' not in test_gdf_full.columns or test_gdf_full['class_id'].notna().sum() == 0:
                            logger.info(f"Skipping {fp}: no labeled data found")
                            continue
                    
                    # Convert GPKG to Parquet for efficient processing
                    logger.info(f"Converting labeled confidence file to Parquet: {fp}")
                    try:
                        parquet_path = convert_zonal_stats_gpkg_to_parquet(fp)
                        logger.info(f"✅ Converted to Parquet: {parquet_path}")
                        # Read from Parquet instead of GPKG
                        df = pd.read_parquet(parquet_path)
                        # Convert back to GeoDataFrame if geometry is needed, otherwise use DataFrame
                        if 'geometry' in df.columns:
                            gdf = gpd.GeoDataFrame(df, geometry='geometry')
                        else:
                            # If no geometry, create a dummy GeoDataFrame or use DataFrame directly
                            gdf = gpd.GeoDataFrame(df)
                    except Exception as conv_err:
                        logger.warning(f"Failed to convert {fp} to Parquet: {conv_err}. Reading GPKG directly.")
                        # Fallback to reading GPKG directly
                    try:
                        gdf = gpd.read_file(fp, where="class_id IS NOT NULL")
                    except Exception as _:
                        # Fallback to full read if driver doesn't support 'where'
                        gdf_full = gpd.read_file(fp)
                        if 'class_id' in gdf_full.columns:
                            gdf = gdf_full.dropna(subset=['class_id'])
                        else:
                            logger.info(f"Skipping {fp}: no class_id column")
                            continue
                    
                    # Final verification
                    if 'class_id' not in gdf.columns:
                        logger.info(f"Skipping {fp}: no class_id column in full file")
                        continue
                    
                    non_null_count = gdf['class_id'].notna().sum()
                    if non_null_count == 0:
                        logger.info(f"Skipping {fp}: no class_id values found in full file")
                        continue
                    
                    # Log columns and class_id stats per file
                    logger.info(f"Read file: {fp} | rows={len(gdf)} | columns={list(gdf.columns)}")
                    
                    # Get unique values for logging
                    unique_vals = sorted([v for v in gdf['class_id'].dropna().unique()])
                    logger.info(f"class_id present | non-null={non_null_count} | unique={unique_vals}")
                    
                    # Keep only essential columns to minimize memory
                    essential_cols = set(feature_columns + ['class_id', 'geometry'])
                    keep_cols = [c for c in gdf.columns if c in essential_cols]
                    gdf = gdf[keep_cols]
                    # Add to labeled data since we confirmed it has labels
                    labeled_gdfs.append(gdf)
                    logger.info(f"Added {fp} with {non_null_count} labeled samples")
                except Exception as e:
                    logger.warning(f"Failed to read confidence file {fp}: {e}")
            if labeled_gdfs:
                # Merge labeled data from all files and retrain once
                logger.info("Merging labeled samples across all confidence files for a single retraining run...")
                merged_labeled_parts = []
                total_samples = 0
                for i, gdf in enumerate(labeled_gdfs):
                    try:
                        # Ensure CRS is consistent
                        target_crs = 'EPSG:4326'
                        if gdf.crs != target_crs:
                            logger.info(f"Converting CRS for file {i+1} from {gdf.crs} to {target_crs}")
                            gdf = gdf.to_crs(target_crs)
                        # Keep only labeled rows and essential columns
                        labeled_data = gdf.dropna(subset=['class_id'])
                        if not labeled_data.empty:
                            merged_labeled_parts.append(labeled_data)
                            total_samples += len(labeled_data)
                            logger.info(f"File {i+1}: added {len(labeled_data)} labeled samples")
                        else:
                            logger.info(f"File {i+1}: no labeled samples; skipped")
                    except Exception as e:
                        logger.warning(f"Failed to prepare labeled data from file {i+1}: {e}")
                
                if not merged_labeled_parts:
                    logger.warning("⚠️  No labeled samples found across files; skipping retraining")
                else:
                    merged_labeled_df = pd.concat(merged_labeled_parts, ignore_index=True)
                    logger.info("=" * 80)
                    logger.info("📊 MERGED LABELED DATASET SUMMARY")
                    logger.info("=" * 80)
                    logger.info(f"Total labeled samples before filtering: {len(merged_labeled_df):,}")
                    
                    # Restrict to valid classes if needed
                    merged_labeled_df = merged_labeled_df[merged_labeled_df['class_id'].isin([0, 1, 2])]
                    logger.info(f"Total labeled samples after class filter (class_id in [0,1,2]): {len(merged_labeled_df):,}")
                    
                    if len(merged_labeled_df) > 0:
                        class_dist = merged_labeled_df['class_id'].value_counts().sort_index()
                        logger.info("Class distribution in merged dataset:")
                        for class_id, count in class_dist.items():
                            class_name = CLASS_NAMES.get(class_id, f"Unknown({class_id})")
                            pct = (count / len(merged_labeled_df) * 100) if len(merged_labeled_df) > 0 else 0
                            logger.info(f"  - {class_name} (class_id={class_id}): {count:,} samples ({pct:.1f}%)")
                    logger.info("=" * 80)

                    # Access base PyTorch model
                    if hasattr(clf, 'pytorch_model'):
                        base_model = clf.pytorch_model
                    else:
                        base_model = getattr(clf, 'pytorch_model', None)
                    if base_model is None:
                        logger.error("Cannot access PyTorch model for retraining. Please ensure model is properly loaded.")
                    else:
                        # Ensure attributes are present
                        if not hasattr(base_model, 'feature_columns'):
                            base_model.feature_columns = feature_columns
                        if not hasattr(base_model, 'scaler') and scaler is not None:
                            base_model.scaler = scaler
                            logger.info("Attached scaler to base_model for retraining")

                        # Retrain once on merged dataset
                        logger.info(f"Retraining on merged dataset with {len(merged_labeled_df)} samples...")
                        updated_model = retrain_model(base_model, merged_labeled_df, config)
                        # Update the predictor
                        if hasattr(clf, 'pytorch_model'):
                            clf.pytorch_model = updated_model
                        else:
                            clf = PyTorchPredictor(updated_model, feature_columns)
                        retraining_occurred = True
                        _retraining_already_occurred = True  # Set module-level flag
                        logger.info("Completed retraining on merged labeled dataset")
                        logger.info("✅ Retraining flag set - subsequent calls will skip retraining")
            else:
                logger.info("No readable labeled confidence files found; skipping retraining")
        except Exception as e:
            logger.error(f"Error during retraining flow: {e}")
    
    # Note: We check per-file whether predictions exist in process_single_file()
    # This allows us to skip files that already have predictions while processing missing ones
    if not retraining_occurred:
        logger.info("No retraining occurred. Will check per-file if predictions exist and skip those files...")
    
    # Now make predictions on all files with the retrained model
    logger.info("Making predictions on all files with retrained model...")
    
    supported_extensions = (".gpkg", ".parquet")
    if isinstance(zonal_stats_dir, str) and zonal_stats_dir.startswith("s3://"):
        raise ValueError("S3 zonal statistics directories are no longer supported. Please use a local directory.")
    zonal_files = []
    for pattern in ("*.gpkg", "*.parquet"):
        zonal_files.extend(glob.glob(os.path.join(zonal_stats_dir, pattern)))
    zonal_files.sort()
    logger.info(f"Found {len(zonal_files)} zonal stats files locally")
    
    # Check for labels in GPKG files and convert to Parquet if labels are found
    from treelance_sentinel.imagery_processing import convert_zonal_stats_gpkg_to_parquet
    
    logger.info("Checking zonal stats GPKG files for labels (class_id)...")
    for zonal_file in zonal_files:
        try:
            check_path = zonal_file
            
            # Check if class_id exists and has non-null values
            # Use OGR SQL filter to efficiently check for non-null class_id
            has_labels = False
            non_null_count = 0
            
            try:
                # Try to read only rows with non-null class_id (efficient)
                gdf_labeled = gpd.read_file(check_path, where="class_id IS NOT NULL")
                if not gdf_labeled.empty and 'class_id' in gdf_labeled.columns:
                    has_labels = True
                    non_null_count = len(gdf_labeled)
                    logger.info(f"Found labels in {os.path.basename(zonal_file)}: {non_null_count} labeled samples")
            except Exception:
                # Fallback: read full file and check
                try:
                    gdf_full = gpd.read_file(check_path)
                    if 'class_id' in gdf_full.columns and gdf_full['class_id'].notna().any():
                        has_labels = True
                        non_null_count = gdf_full['class_id'].notna().sum()
                        logger.info(f"Found labels in {os.path.basename(zonal_file)}: {non_null_count} labeled samples")
                except Exception as read_err:
                    logger.debug(f"Could not read {check_path} to check for labels: {read_err}")
            
            # Convert to Parquet if labels are found
            if has_labels:
                logger.info(f"Converting labeled GPKG to Parquet: {zonal_file}")
                try:
                    parquet_path = convert_zonal_stats_gpkg_to_parquet(zonal_file)
                    logger.info(f"✅ Successfully converted to Parquet: {parquet_path}")
                except Exception as conv_err:
                    logger.warning(f"Failed to convert {zonal_file} to Parquet: {conv_err}")
            
        except Exception as e:
            logger.warning(f"Error checking labels in {zonal_file}: {e}")
            continue
    
    # Limit parallel workers to 1-2 to reduce GPU/CPU memory pressure
    max_workers = int(config.get('prediction', {}).get('max_workers', 1))
    logger.info(f"Using {max_workers} workers for parallel processing")

    # Process files in parallel using global executor
    global _global_executor
    try:
        _global_executor = ThreadPoolExecutor(max_workers=max_workers)
        
        try:
            from tqdm import tqdm
            progress = tqdm(total=len(zonal_files), desc="Predicting", unit="file")
        except Exception:
            progress = None

        # Create futures for all files
        future_to_file = {}
        for zonal_file in zonal_files:
            future = _global_executor.submit(
                process_single_file,
                zonal_file,
                model,
                feature_columns,
                config,
                output_dirs,
                clf,
                scaler
            )
            future_to_file[future] = zonal_file

        logger.info(f"Submitted {len(future_to_file)} files for parallel processing")
        
        # Process all futures
        completed_count = 0
        for future in as_completed(future_to_file):
            # Periodic GPU cleanup every 5 files to prevent memory buildup
            if completed_count > 0 and completed_count % 5 == 0:
                cleanup_gpu()
            zonal_file = future_to_file[future]  # Get the file path for this future
            completed_count += 1
            try:
                future.result()
                logger.info(f"✅ [{completed_count}/{len(zonal_files)}] Successfully processed: {os.path.basename(zonal_file)}")
            except Exception as e:
                logger.error(f"❌ [{completed_count}/{len(zonal_files)}] Error processing {os.path.basename(zonal_file)}: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
            finally:
                if progress:
                    progress.update(1)
                # Cleanup GPU after each file to prevent memory buildup
                cleanup_gpu()
        
        if progress:
            progress.close()
        
        logger.info(f"✅ Completed processing {completed_count}/{len(zonal_files)} zonal statistics files")
                
    finally:
        # Ensure executor is properly shutdown
        if _global_executor is not None:
            _global_executor.shutdown(wait=True, cancel_futures=True)
            _global_executor = None
        # Final GPU cleanup
        cleanup_gpu()

def generate_class_summary(predictions_dir: str) -> None:
    """
    Generate a summary of class percentages from all prediction files.
    
    Args:
        predictions_dir: Directory containing prediction results
    """
    logger.info("\n" + "="*60)
    logger.info("CLASSIFICATION SUMMARY")
    logger.info("="*60)
    
    # Find all prediction files
    if isinstance(predictions_dir, str) and predictions_dir.startswith("s3://"):
        raise ValueError("S3 prediction directories are no longer supported. Please use a local directory.")
    raw_dir = os.path.join(predictions_dir, "raw")
    if not os.path.exists(raw_dir):
        logger.warning(f"Raw predictions directory not found: {raw_dir}")
        return
    prediction_files = glob.glob(os.path.join(raw_dir, "*_predicted.gpkg"))
    logger.info(f"Found {len(prediction_files)} prediction files locally")
    
    if not prediction_files:
        logger.warning("No prediction files found")
        return
    
    # Collect all predictions
    all_predictions = []
    total_area = 0
    
    for file_path in prediction_files:
        try:
            gdf = gpd.read_file(file_path)
            
            if not gdf.empty:
                # Ensure we have the right CRS for area calculation
                if gdf.crs is None:
                    gdf = gdf.set_crs('EPSG:4326')
                
                # Convert to Web Mercator for accurate area calculation
                gdf_area = gdf.to_crs('EPSG:3857')
                gdf['area_sq_m'] = gdf_area.geometry.area
                
                # Debug: Check area calculation
                logger.debug(f"File {os.path.basename(file_path)}: total area = {gdf['area_sq_m'].sum():.2f} sq m")
                logger.debug(f"Class distribution: {gdf['predicted_class'].value_counts().to_dict()}")
                
                all_predictions.append(gdf)
                total_area += gdf['area_sq_m'].sum()
                
                logger.info(f"Loaded {len(gdf)} predictions from {os.path.basename(file_path)}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    if not all_predictions:
        logger.warning("No valid prediction data found")
        return
    
    # Combine all predictions - ensure consistent CRS first
    # Convert all to a common CRS (WGS84) before concatenation
    target_crs = 'EPSG:4326'
    normalized_predictions = []
    for gdf in all_predictions:
        if gdf.crs is None:
            gdf = gdf.set_crs(target_crs)
        elif gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)
        normalized_predictions.append(gdf)
    
    combined_gdf = pd.concat(normalized_predictions, ignore_index=True)
    
    # Calculate class statistics
    class_stats = {}
    total_predictions = len(combined_gdf)
    
    for class_id, class_name in CLASS_NAMES.items():
        class_mask = combined_gdf['predicted_class'] == class_id
        class_count = class_mask.sum()
        class_area = combined_gdf.loc[class_mask, 'area_sq_m'].sum()
        
        percentage_count = (class_count / total_predictions * 100) if total_predictions > 0 else 0
        percentage_area = (class_area / total_area * 100) if total_area > 0 else 0
        
        class_stats[class_name] = {
            'count': class_count,
            'area_sq_m': class_area,
            'area_sq_km': class_area / 1_000_000,  # Convert to km²
            'percentage_count': percentage_count,
            'percentage_area': percentage_area
        }
        
        # Debug logging
        logger.debug(f"Class {class_name}: count={class_count}, area={class_area}, percentage_area={percentage_area}")
    
    # Print summary
    logger.info(f"\nTotal predictions: {total_predictions:,}")
    logger.info(f"Total area: {total_area / 1_000_000:.2f} km²")
    logger.info("\nClass Distribution:")
    logger.info("-" * 80)
    logger.info(f"{'Class':<15} {'Count':<10} {'Area (km²)':<12} {'% Count':<10} {'% Area':<10}")
    logger.info("-" * 80)
    
    for class_name, stats in class_stats.items():
        logger.info(
            f"{class_name:<15} "
            f"{stats['count']:<10,} "
            f"{stats['area_sq_km']:<12.2f} "
            f"{stats['percentage_count']:<10.2f} "
            f"{stats['percentage_area']:<10.2f}"
        )
    
    logger.info("-" * 80)
    
    # Save summary to file
    summary_file = os.path.join(predictions_dir, "classification_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("CLASSIFICATION SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total predictions: {total_predictions:,}\n")
        f.write(f"Total area: {total_area / 1_000_000:.2f} km²\n")
        f.write("\nClass Distribution:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<15} {'Count':<10} {'Area (km²)':<12} {'% Count':<10} {'% Area':<10}\n")
        f.write("-" * 80 + "\n")
        
        for class_name, stats in class_stats.items():
            f.write(
                f"{class_name:<15} "
                f"{stats['count']:<10,} "
                f"{stats['area_sq_km']:<12.2f} "
                f"{stats['percentage_count']:<10.2f} "
                f"{stats['percentage_area']:<10.2f}\n"
            )
        
        f.write("-" * 80 + "\n")
    
    logger.info(f"\nSummary saved to: {summary_file}")
    logger.info("="*60)

@timer_decorator
def main(
    zonal_stats_dir: str,
    predictions_dir: str,
    model_path: str,
    config: dict,
    retraining_confidence_dir: str | None = None,
) -> None:
    """Main function for model prediction."""
    global _global_model

    try:
        logger.info("Starting prediction pipeline...")

        # Load the PyTorch model
        logger.info(f"Loading model from {model_path}")
        model = load_pytorch_model(model_path)
        _global_model = model  # Store for cleanup

        # Make predictions
        logger.info("Making predictions...")
        make_predictions(
            model,
            zonal_stats_dir,
            predictions_dir,
            config,
            retraining_confidence_dir=retraining_confidence_dir,
        )
        
        # Generate and save class summary
        generate_class_summary(predictions_dir)
        
        logger.info("Prediction pipeline completed successfully.")
        
    except Exception as e:
        logger.error(f"Error in prediction pipeline: {e}")
        raise
    finally:
        # Ensure cleanup happens even if there's an error
        logger.info("Cleaning up resources...")
        cleanup_resources()
        # Final GPU cleanup check
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()

if __name__ == '__main__':
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--zonal_stats_dir', required=True)
    parser.add_argument('--predictions_dir', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--config', required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Pass all required arguments to main
    main(args.zonal_stats_dir, args.predictions_dir, args.model_path, config) 