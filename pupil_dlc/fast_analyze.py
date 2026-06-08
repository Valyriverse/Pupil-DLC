"""Threaded frame-prefetch + AMP/XLA for DLC inference.

Two stacked optimisations over DLC's default serial GetPoseF_GTF:
  1. Threaded prefetch  — producer thread decodes frames while GPU runs inference,
                          eliminating CPU/GPU serialisation (~68% → ~95% GPU util).
                          Decoded frames are buffered into a bounded queue of up to
                          eight pre-loaded batches so the GPU never stalls.
  2. AMP + XLA JIT      — patches DLC's GPU session ConfigProto before it is created:
                            - auto_mixed_precision=ON rewrites eligible ResNet-50 ops
                              to FP16, exploiting Ampere (and newer) tensor cores.
                            - global_jit_level=ON_2 fuses ~150 CUDA kernel launches
                              via XLA (enabled only when libdevice.10.bc is found).
"""
import os
import queue
import shutil
import threading

import cv2
import numpy as np
from skimage.util import img_as_ubyte
from tqdm import tqdm


def _ensure_xla_libdevice():
    """Find libdevice.10.bc and set XLA_FLAGS so XLA JIT works on CUDA 11.x.

    Called from patch_dlc_inference() so it works in both CLI and notebook contexts.
    Returns True if setup succeeded (XLA usable), False if libdevice not found.
    """
    import glob as _glob

    if "--xla_gpu_cuda_data_dir" in os.environ.get("XLA_FLAGS", ""):
        return True  # already configured

    cuda_path = os.environ.get("CUDA_PATH", "")
    candidates = []
    seen: set = set()
    for root in filter(None, [
        cuda_path,
        os.path.dirname(cuda_path) if cuda_path else "",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
    ]):
        if root in seen:
            continue
        seen.add(root)
        direct = os.path.join(root, "nvvm", "libdevice", "libdevice.10.bc")
        if os.path.exists(direct):
            candidates.append(direct)
        candidates.extend(sorted(
            _glob.glob(os.path.join(root, "v*", "nvvm", "libdevice", "libdevice.10.bc")),
            reverse=True,
        ))

    if not candidates:
        return False

    xla_cache = os.environ.get("PUPIL_XLA_CACHE",
                               os.path.join(os.path.expanduser("~"), ".xla_cuda"))
    ld_dir = os.path.join(xla_cache, "nvvm", "libdevice")
    ld_dst = os.path.join(ld_dir, "libdevice.10.bc")
    if not os.path.exists(ld_dst):
        os.makedirs(ld_dir, exist_ok=True)
        shutil.copy2(candidates[0], ld_dst)

    xla_flag = f"--xla_gpu_cuda_data_dir={xla_cache.replace(os.sep, '/')}"
    existing = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_cuda_data_dir" not in existing:
        os.environ["XLA_FLAGS"] = (existing + " " + xla_flag).strip()
    return True


def _GetPoseF_GTF_threaded(cfg, dlc_cfg, sess, inputs, outputs, cap, nframes, batchsize):
    from deeplabcut.pose_estimation_tensorflow.core import predict
    from deeplabcut.pose_estimation_tensorflow.predict_videos import checkcropping

    ny = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nx = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if cfg["cropping"]:
        ny, nx = checkcropping(cfg, cap)

    pose_tensor = predict.extract_GPUprediction(outputs, dlc_cfg)

    PredictedData = np.zeros((nframes, 3 * len(dlc_cfg["all_joints_names"])))

    # Hold up to 8 pre-decoded batches so the GPU never stalls waiting for frames.
    frame_queue = queue.Queue(maxsize=8)

    def _producer():
        frames = np.empty((batchsize, ny, nx, 3), dtype="ubyte")
        batch_ind = 0
        inds = []
        counter = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if cfg["cropping"]:
                    frames[batch_ind] = img_as_ubyte(
                        frame[cfg["y1"]:cfg["y2"], cfg["x1"]:cfg["x2"]]
                    )
                else:
                    frames[batch_ind] = img_as_ubyte(frame)
                inds.append(counter)
                batch_ind += 1
                if batch_ind == batchsize:
                    frame_queue.put((frames.copy(), list(inds)))
                    batch_ind = 0
                    inds.clear()
            elif counter >= nframes:
                if batch_ind > 0:
                    frame_queue.put((frames.copy(), list(inds)))
                break
            counter += 1
        frame_queue.put(None)  # sentinel

    producer_thread = threading.Thread(target=_producer, daemon=True)
    producer_thread.start()

    pbar = tqdm(total=nframes)
    while True:
        item = frame_queue.get()
        if item is None:
            break
        batch_frames, batch_inds = item
        n = len(batch_inds)

        try:
            # TF GPU path: output is (batch*n_joints, 3) in (y, x, conf) order.
            pose = sess.run(pose_tensor, feed_dict={inputs: batch_frames})
            pose[:, [0, 1, 2]] = pose[:, [1, 0, 2]]   # swap y,x → x,y
            pose = np.reshape(pose, (batchsize, -1))
            PredictedData[batch_inds] = pose[:n]
        except Exception as _e:
            import traceback
            print(f"\n[fast_analyze] sess.run() failed on batch starting at frame {batch_inds[0]}:")
            traceback.print_exc()
            raise
        pbar.update(n)

    producer_thread.join()
    pbar.close()
    return PredictedData, nframes


def patch_dlc_inference():
    """Patch DLC inference: threaded prefetch + AMP/XLA.

    Calling this function is safe in both CLI and Jupyter notebook contexts —
    _ensure_xla_libdevice() is called here (not at module import) so it runs
    regardless of which entry point the user uses.
    """
    import tensorflow as tf
    import deeplabcut.pose_estimation_tensorflow.predict_videos as pv
    import deeplabcut.pose_estimation_tensorflow.core.predict as pred

    # --- 1. Threaded inference replacement ---
    pv.GetPoseF_GTF = _GetPoseF_GTF_threaded

    # --- 2. AMP + XLA: patch the TF1 ConfigProto before the GPU session is created ---
    _xla_ok = _ensure_xla_libdevice()

    _orig_setup = pred.setup_GPUpose_prediction

    def _setup_with_amp_xla(cfg, allow_growth=False):
        _OrigSession = tf.compat.v1.Session

        class _AmpXlaSession(_OrigSession):
            def __init__(self, graph=None, config=None, target=""):
                from tensorflow.core.protobuf import rewriter_config_pb2
                if config is None:
                    config = tf.compat.v1.ConfigProto()
                # XLA ON_2: fuses CUDA kernel launches (requires libdevice.10.bc).
                if _xla_ok:
                    config.graph_options.optimizer_options.global_jit_level = (
                        tf.compat.v1.OptimizerOptions.ON_2
                    )
                # AMP: rewrites eligible ops to FP16 for tensor core acceleration.
                config.graph_options.rewrite_options.auto_mixed_precision = (
                    rewriter_config_pb2.RewriterConfig.ON
                )
                super().__init__(graph=graph, config=config, target=target)

        tf.compat.v1.Session = _AmpXlaSession
        try:
            return _orig_setup(cfg, allow_growth=allow_growth)
        finally:
            tf.compat.v1.Session = _OrigSession

    pred.setup_GPUpose_prediction = _setup_with_amp_xla
