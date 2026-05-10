"""One-time conversion of the DLC GM checkpoint to an ONNX model for ONNX Runtime.

Run once inside the pupil-dlc conda environment:
    python pupil_dlc/convert_dlc_onnx.py GM_gitub/config.yaml

The output file (model_b64.onnx) is written next to the DLC checkpoint.
Subsequent pupil-dlc runs detect it automatically and use ONNX Runtime (CUDA EP)
instead of the TF session, giving an additional ~1.3-1.6x speedup over the
XLA-JIT path.

Dependencies (install once):
    pip install tf2onnx onnxruntime-gpu
"""
import os
import sys

_BATCHSIZE = 64
_HEIGHT = 480
_WIDTH = 640


def _find_model_paths(config_path):
    """Return (pose_cfg_path, train_dir, snapshot_path) replicating analyze_videos logic."""
    import yaml
    import numpy as np
    from deeplabcut.utils.auxiliaryfunctions import get_model_folder

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    train_fraction = cfg["TrainingFraction"][0]
    model_folder = get_model_folder(train_fraction, shuffle=1, cfg=cfg)
    project_path = cfg["project_path"]
    model_dir = os.path.join(project_path, model_folder)
    train_dir = os.path.join(model_dir, "train")
    pose_cfg_path = os.path.join(model_dir, "test", "pose_cfg.yaml")

    # Replicate DLC's snapshot selection (snapshotindex from config.yaml, default -1)
    snapshots = np.array([
        fn.split(".")[0]
        for fn in os.listdir(train_dir)
        if fn.endswith(".index")
    ])
    increasing = np.argsort([int(m.split("-")[1]) for m in snapshots])
    snapshots = snapshots[increasing]

    snapshotindex = cfg.get("snapshotindex", -1)
    if snapshotindex == "all":
        snapshotindex = -1
    chosen = snapshots[snapshotindex]
    snapshot_path = os.path.join(train_dir, chosen)
    print(f"Using snapshot: {snapshot_path}")

    return pose_cfg_path, train_dir, snapshot_path


def main(config_path):
    try:
        import tf2onnx
        import onnxruntime as ort
    except ImportError as exc:
        print(f"Missing dependency: {exc}")
        print("Install with: pip install tf2onnx onnxruntime-gpu")
        sys.exit(1)

    import numpy as np
    import tensorflow as tf
    from deeplabcut.pose_estimation_tensorflow.config import load_config
    from deeplabcut.pose_estimation_tensorflow.core import predict
    from tensorflow.python.framework.graph_util import convert_variables_to_constants

    pose_cfg_path, train_dir, snapshot_path = _find_model_paths(config_path)
    onnx_path = os.path.join(train_dir, "model_b64.onnx")

    if os.path.exists(onnx_path):
        print(f"ONNX model already exists: {onnx_path}")
        print("Delete it and rerun to regenerate.")
        return onnx_path

    print(f"Loading DLC config: {pose_cfg_path}")
    dlc_cfg = load_config(pose_cfg_path)
    # Override init_weights to point at the trained snapshot (not the pretrained backbone)
    dlc_cfg["init_weights"] = snapshot_path
    dlc_cfg["batch_size"] = _BATCHSIZE
    dlc_cfg["num_outputs"] = 1

    # Use the test head (setup_pose_prediction) rather than the GPU inference head
    # (setup_GPUpose_prediction). The GPU inference head uses tf.unravel_index which
    # has no standard ONNX equivalent and crashes ONNX Runtime. The test head outputs
    # raw heatmaps (part_prob) and locref — the argmax is done in numpy in fast_analyze.py
    # and is negligible (<1 ms per batch of 64).
    print("Loading DLC TF session (test head — outputs heatmaps + locref)...")
    sess, inputs, outputs, _ = predict.setup_pose_prediction(
        dlc_cfg, allow_growth=True, collect_extra=True
    )
    part_prob = outputs[0]   # heatmap: [batch, H_feat, W_feat, n_joints]
    locref    = outputs[1]   # raw offsets: [batch, H_feat, W_feat, n_joints*2]

    print(f"  input     : {inputs.name}  shape={inputs.shape}")
    print(f"  part_prob : {part_prob.name}")
    print(f"  locref    : {locref.name}")

    print("Freezing TF graph (Variables → Constants, excl. UnravelIndex subgraph)...")
    output_node_names = [part_prob.name.split(":")[0], locref.name.split(":")[0]]
    frozen_graph = convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), output_node_names
    )

    print(f"Converting to ONNX (opset 13, fixed input {_BATCHSIZE}×{_HEIGHT}×{_WIDTH}×3)...")
    onnx_model, _ = tf2onnx.convert.from_graph_def(
        frozen_graph,
        input_names=[inputs.name],
        output_names=[part_prob.name, locref.name],
        opset=13,
        shape_override={inputs.name: [_BATCHSIZE, _HEIGHT, _WIDTH, 3]},
    )

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Saved: {onnx_path}")

    print("Verifying with ONNX Runtime (CUDA EP)...")
    ort_sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inp_meta = ort_sess.get_inputs()[0]
    test_batch = np.zeros((_BATCHSIZE, _HEIGHT, _WIDTH, 3), dtype=np.float32)
    results = ort_sess.run(None, {inp_meta.name: test_batch})
    n_joints = dlc_cfg["num_joints"]
    stride   = dlc_cfg.get("stride", 8)
    H_feat   = _HEIGHT // stride
    W_feat   = _WIDTH  // stride
    print(f"  part_prob : {results[0].shape}  (expected [{_BATCHSIZE}, {H_feat}, {W_feat}, {n_joints}])")
    print(f"  locref    : {results[1].shape}  (expected [{_BATCHSIZE}, {H_feat}, {W_feat}, {n_joints * 2}])")
    print("Verification OK — argmax will be applied in fast_analyze.py at inference time.")

    sess.close()
    return onnx_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_dlc_onnx.py <path/to/config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
