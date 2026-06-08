###################### Pupil-DLC Pipeline#########################

#!/usr/bin/env python
import os
import sys
import shutil as _shutil
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("TF_GPU_THREAD_MODE", "gpu_private")
# XLA_FLAGS splits on spaces, so "Program Files" breaks the path.
# Copy libdevice once to a space-free location and point XLA there.
# Searches installed CUDA versions automatically; override with PUPIL_XLA_CACHE env var.
def _setup_xla_libdevice():
    import glob as _glob

    _candidates = []
    _cuda_path = os.environ.get("CUDA_PATH", "")

    # CUDA_PATH on Windows is often already versioned (e.g. C:\...\CUDA\v11.2).
    # Check it directly first, then also glob the parent for all installed versions.
    _seen_roots = set()
    for _root in filter(None, [
        _cuda_path,                                          # versioned direct path
        os.path.dirname(_cuda_path) if _cuda_path else "",  # parent of versioned path
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",  # fallback
    ]):
        if _root in _seen_roots:
            continue
        _seen_roots.add(_root)
        _direct = os.path.join(_root, "nvvm", "libdevice", "libdevice.10.bc")
        if os.path.exists(_direct):
            _candidates.append(_direct)
        _candidates.extend(sorted(
            _glob.glob(os.path.join(_root, "v*", "nvvm", "libdevice", "libdevice.10.bc")),
            reverse=True
        ))

    if not _candidates:
        return  # no CUDA found; skip XLA setup

    _ld_src = _candidates[0]
    _xla_cache = os.environ.get("PUPIL_XLA_CACHE", os.path.join(os.path.expanduser("~"), ".xla_cuda"))
    _ld_dir = os.path.join(_xla_cache, "nvvm", "libdevice")
    _ld_dst = os.path.join(_ld_dir, "libdevice.10.bc")

    if not os.path.exists(_ld_dst):
        os.makedirs(_ld_dir, exist_ok=True)
        _shutil.copy2(_ld_src, _ld_dst)
    _xla_flag = f"--xla_gpu_cuda_data_dir={_xla_cache.replace(os.sep, '/')}"
    existing = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_cuda_data_dir" not in existing:
        os.environ["XLA_FLAGS"] = (existing + " " + _xla_flag).strip()

_setup_xla_libdevice()
#import matplotlib
#matplotlib.use('Agg')
import click
import pyfiglet
import pandas as pd
import deeplabcut
import fnmatch
import time
from tqdm import tqdm

##### Pupil reinstallment should be in the main folder where the setup.py is and do pip install -e .
from .smoothing_module import smooth_pupil_data, filter_by_rate_of_change
from .ellipse import ellipse_fitting
from .yaml_section import replace_yaml_section
from .fast_analyze import patch_dlc_inference

# Two-part split archive from https://doi.org/10.6084/m9.figshare.31282714
# Each entry: (figshare_file_id, local_7z_filename, label_for_display)
_FIGSHARE_FILES = [
    ("65291688", "Labeled-data-1st.7z", "part 1 (~6.2 GB)"),
    ("65291697", "labeled-data-2nd.7z", "part 2 (~6.3 GB)"),
]
_GLOBAL_GM_CACHE = os.path.join(os.path.expanduser("~"), ".pupil_dlc_cache", "gm_labeled_data")
_GM_SCORER = "Parsa"  # scorer name embedded in the Figshare dataset
# Bump this when the Figshare dataset changes so stale caches are wiped and re-downloaded.
_CACHE_VERSION = "2"


def _patch_config_project_path(config_path):
    """Rewrite project_path in config.yaml to match config file's actual location."""
    import yaml
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    actual = os.path.normpath(os.path.dirname(config_path))
    if os.path.normpath(cfg.get('project_path', '')) != actual:
        cfg['project_path'] = actual
        with open(config_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        click.echo(f"→ updated project_path in config to: {actual}")
    return actual


def _ensure_gm_cache():
    """Download and extract GM labeled data (two-part 7z) to a global cache — runs once ever."""
    import requests
    import py7zr

    cache_marker = os.path.join(_GLOBAL_GM_CACHE, ".gm_data_ok")
    if os.path.exists(cache_marker):
        cached_version = open(cache_marker).read().strip()
        if cached_version == _CACHE_VERSION:
            return  # already cached at current version
        click.secho(
            f"→ GM cache version mismatch (have {cached_version!r}, need {_CACHE_VERSION!r}) "
            f"— wiping and re-downloading…", fg="yellow"
        )
        _shutil.rmtree(_GLOBAL_GM_CACHE)

    os.makedirs(_GLOBAL_GM_CACHE, exist_ok=True)
    cache_parent = os.path.dirname(_GLOBAL_GM_CACHE)

    for file_id, archive_name, part_label in _FIGSHARE_FILES:
        url = f"https://ndownloader.figshare.com/files/{file_id}"
        archive_path = os.path.join(cache_parent, archive_name)
        click.secho(
            f"→ downloading GM labeled data {part_label} from Figshare — one-time download, "
            f"cached at {_GLOBAL_GM_CACHE}…", fg="yellow"
        )
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        total = int(resp.headers.get('content-length', 0))
        with open(archive_path, 'wb') as f, tqdm(
            total=total, unit='B', unit_scale=True, desc=archive_name
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))
        click.echo(f"→ {part_label} download complete, extracting…")

        # Extract into a temp subfolder then flatten one top-level directory if present
        tmp_dir = archive_path + "_extracted"
        os.makedirs(tmp_dir, exist_ok=True)
        with py7zr.SevenZipFile(archive_path, mode='r') as zf:
            zf.extractall(path=tmp_dir)

        # If the archive extracted into a single top-level folder, descend into it
        entries = os.listdir(tmp_dir)
        src_root = os.path.join(tmp_dir, entries[0]) if (
            len(entries) == 1 and os.path.isdir(os.path.join(tmp_dir, entries[0]))
        ) else tmp_dir

        for item in tqdm(os.listdir(src_root), desc=f"staging {part_label}"):
            _shutil.move(os.path.join(src_root, item), os.path.join(_GLOBAL_GM_CACHE, item))

        _shutil.rmtree(tmp_dir)
        os.remove(archive_path)

    open(cache_marker, 'w').write(_CACHE_VERSION)
    click.secho(f"→ GM data cached globally at {_GLOBAL_GM_CACHE}", fg="green")


def _link_gm_data_to_project(project_dir, scorer):
    """Hard-link GM images into the project (zero disk cost) and copy annotation
    files with the scorer name updated to match the new project's scorer."""
    labeled_data_dir = os.path.join(project_dir, "labeled-data")
    marker = os.path.join(labeled_data_dir, ".gm_data_ok")
    if os.path.exists(marker):
        click.echo("→ GM labeled-data already linked to this project, skipping.")
        return

    session_dirs = [
        d for d in os.listdir(_GLOBAL_GM_CACHE)
        if os.path.isdir(os.path.join(_GLOBAL_GM_CACHE, d)) and not d.startswith('.')
    ]
    click.echo(f"→ linking {len(session_dirs)} GM sessions into project (scorer → {scorer})…")

    for session in tqdm(session_dirs, desc="linking"):
        src_session = os.path.join(_GLOBAL_GM_CACHE, session)
        dst_session = os.path.join(labeled_data_dir, session)
        os.makedirs(dst_session, exist_ok=True)

        for fname in os.listdir(src_session):
            src_file = os.path.join(src_session, fname)
            if os.path.isdir(src_file):
                continue

            if fname == f"CollectedData_{_GM_SCORER}.h5":
                # Copy H5 with internal scorer MultiIndex level renamed
                dst_file = os.path.join(dst_session, f"CollectedData_{scorer}.h5")
                if not os.path.exists(dst_file):
                    df = pd.read_hdf(src_file, 'df_with_missing')
                    df = df.rename(columns={_GM_SCORER: scorer}, level='scorer')
                    df.to_hdf(dst_file, key='df_with_missing', mode='w')

            elif fname == f"CollectedData_{_GM_SCORER}.csv":
                # Copy CSV with scorer header row updated
                dst_file = os.path.join(dst_session, f"CollectedData_{scorer}.csv")
                if not os.path.exists(dst_file):
                    df = pd.read_csv(src_file, header=[0, 1, 2], index_col=0)
                    df = df.rename(columns={_GM_SCORER: scorer}, level=0)
                    df.to_csv(dst_file)

            else:
                # PNG images — hard-link (zero extra disk space, instant);
                # fall back to copy if project is on a different drive than cache.
                dst_file = os.path.join(dst_session, fname)
                if not os.path.exists(dst_file):
                    try:
                        os.link(src_file, dst_file)
                    except OSError:
                        _shutil.copy2(src_file, dst_file)

    # Register every GM session in video_sets so create_training_dataset finds them.
    # DLC only uses the path string to derive the labeled-data folder stem — the file
    # does not need to exist. Without this, merge_annotateddatasets silently skips
    # any labeled-data folder that has no matching video_sets key.
    import yaml as _yaml
    config_path = os.path.join(project_dir, "config.yaml")
    with open(config_path, 'r') as _f:
        cfg = _yaml.safe_load(_f)
    registered = set(cfg.get("video_sets", {}).keys())
    for session in session_dirs:
        placeholder = os.path.join(project_dir, "videos", f"{session}.mp4")
        if placeholder not in registered:
            cfg.setdefault("video_sets", {})[placeholder] = {"crop": "0, 640, 0, 480"}
    with open(config_path, 'w') as _f:
        _yaml.dump(cfg, _f, default_flow_style=False, allow_unicode=True)

    open(marker, 'w').close()
    click.secho(f"→ GM labeled-data ready in {labeled_data_dir}", fg="green")


def _patch_dlc_labeling_toolbox():
    """Monkey-patch three DLC labeling-toolbox bugs that cause silent data loss and
    progressive slowdown:

    1. saveEachImage — chained indexing silently drops labels.
       Fix: single .loc[(row, col)] tuple indexing.

    2. ImagePanel.drawplot — axes.clear() + imshow() on every navigation accumulates
       matplotlib artists and slows down over time.
       Fix: cache AxesImage in _image_obj; use set_data() on subsequent calls.

    3. mpl_connect accumulation — button_press/release listeners stack up with each
       image navigation because old ones are never disconnected.
       Fix: wrap canvas.mpl_connect to auto-disconnect duplicate event types.
    """
    try:
        import cv2 as _cv2
        import numpy as _np
        from deeplabcut.gui.labeling_toolbox import MainFrame, ImagePanel

        # --- fix 1: saveEachImage ---
        def _saveEachImage_fixed(self):
            for bp in self.updatedCoords:
                self.dataFrame.loc[
                    self.relativeimagenames[self.iter],
                    (self.scorer, bp[0][-2], "x"),
                ] = bp[-1][0]
                self.dataFrame.loc[
                    self.relativeimagenames[self.iter],
                    (self.scorer, bp[0][-2], "y"),
                ] = bp[-1][1]

        MainFrame.saveEachImage = _saveEachImage_fixed

        # --- fix 2: ImagePanel.drawplot — reuse imshow object ---
        def _drawplot_fixed(self, img, img_name, itr, index, bodyparts, cmap, keep_view=False):
            import matplotlib.pyplot as _plt
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar

            xlim = self.axes.get_xlim()
            ylim = self.axes.get_ylim()
            im = _cv2.imread(img)[..., ::-1]
            epLines, sourcePts, offsets = self.retrieveData_and_computeEpLines(img, itr)
            if epLines is not None:
                from deeplabcut.gui.labeling_toolbox import getColorIndices
                norm, colorIndex = getColorIndices(img, bodyparts)
                im = self.drawEpLines(im.copy(), epLines, sourcePts, offsets, colorIndex, cmap, norm)

            if getattr(self, '_image_obj', None) is None:
                self.axes.clear()
                self._image_obj = self.axes.imshow(im, cmap=cmap)
                self.orig_xlim = self.axes.get_xlim()
                self.orig_ylim = self.axes.get_ylim()
                divider = make_axes_locatable(self.axes)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                _plt.colorbar(
                    _plt.cm.ScalarMappable(cmap=cmap),
                    cax=cax,
                    ticks=_np.linspace(0, 1, len(bodyparts)),
                ).set_ticklabels(bodyparts[::-1])
                # Fix 3: install mpl_connect dedup on the canvas the first time drawplot
                # runs. At this point self.canvas is guaranteed to exist (set by BasePanel),
                # and MainFrame will store a reference to the same object, so all
                # subsequent mpl_connect calls from nextImage/prevImage go through here.
                if not getattr(self.canvas, '_pupildlc_dedup', False):
                    _tracked = {}
                    _real_connect = self.canvas.mpl_connect
                    _real_disconnect = self.canvas.mpl_disconnect

                    def _dedup_connect(event, callback):
                        if event in _tracked:
                            try:
                                _real_disconnect(_tracked[event])
                            except Exception:
                                pass
                        cid = _real_connect(event, callback)
                        _tracked[event] = cid
                        return cid

                    self.canvas.mpl_connect = _dedup_connect
                    self.canvas._pupildlc_dedup = True
            else:
                for patch in list(self.axes.patches):
                    patch.remove()
                for txt in list(self.axes.texts):
                    txt.remove()
                self._image_obj.set_data(im)
                if not keep_view:
                    self.axes.set_xlim(self.orig_xlim)
                    self.axes.set_ylim(self.orig_ylim)

            self.axes.set_title(str(itr) + "/" + str(len(index) - 1) + " " + img_name)
            if keep_view:
                self.axes.set_xlim(xlim)
                self.axes.set_ylim(ylim)
            if self.toolbar is None:
                self.toolbar = NavigationToolbar(self.canvas)
            return (self.figure, self.axes, self.canvas, self.toolbar)

        ImagePanel.drawplot = _drawplot_fixed

    except Exception:
        pass  # best-effort; if DLC layout changed, proceed unpatched


def _restore_moved_videos(video_paths, project_dir):
    """If DLC moved videos into project/videos/ (symlink fallback), move them back."""
    videos_dir = os.path.join(project_dir, "videos")
    for vp in video_paths:
        if not os.path.exists(vp):
            moved = os.path.join(videos_dir, os.path.basename(vp))
            if os.path.exists(moved):
                _shutil.move(moved, vp)
                click.secho(f"→ video restored to original location: {vp}", fg="yellow")


def _verify_and_relabel(config_path, project_dir, video_paths, scorer):
    """Check that every user video has saved labels; offer to re-open the GUI if not."""
    _user_label_dirs = [
        os.path.join(project_dir, "labeled-data", os.path.splitext(os.path.basename(vp))[0])
        for vp in video_paths
    ]
    _missing = [
        d for d in _user_label_dirs
        if not os.path.exists(os.path.join(d, f"CollectedData_{scorer}.h5"))
    ]
    if not _missing:
        return True
    click.secho(
        "\nWarning: no saved labels found for your video(s):\n"
        + "\n".join(f"  {d}" for d in _missing)
        + "\n\nMake sure you clicked Save in the labeling GUI before closing it.",
        fg="yellow"
    )
    if not _confirm("Re-open the labeling GUI to save your labels?", default=True):
        click.echo("Proceeding without user labels.")
        return False
    _patch_dlc_labeling_toolbox()
    deeplabcut.label_frames(config_path)
    if not _confirm("Labels saved? Proceed?", default=True):
        click.echo("Aborted.")
        return None  # caller should return
    return True


def _find_gm_snapshot(repo_root):
    """Return the path to the latest snapshot in GM_Model (without .index extension).

    Globs the actual train/ directory so it stays correct after model retraining
    regardless of the DLC project folder name (e.g. GMMay31 vs GMNov17).
    """
    import glob as _glob
    import yaml as _yaml

    gm_dir = os.path.join(repo_root, 'GM_Model')
    gm_config = os.path.join(gm_dir, 'config.yaml')
    if not os.path.exists(gm_config):
        raise click.ClickException(f"GM_Model/config.yaml not found at: {gm_config}")

    index_files = _glob.glob(
        os.path.join(gm_dir, 'dlc-models', 'iteration-0', '*', 'train', 'snapshot-*.index')
    )
    if not index_files:
        raise click.ClickException(
            "No snapshots found in GM_Model/dlc-models/. "
            "Make sure the GM_Model directory is intact."
        )

    with open(gm_config) as _f:
        cfg = _yaml.safe_load(_f)
    snapshotindex = cfg.get('snapshotindex', -1)
    if snapshotindex == 'all':
        snapshotindex = -1

    sorted_snapshots = sorted(
        index_files,
        key=lambda p: int(os.path.basename(p).split('-')[1].split('.')[0])
    )
    chosen = sorted_snapshots[snapshotindex]
    return chosen[:-len('.index')]


def _patch_pose_cfg_for_finetuning(project_dir, gm_snapshot_path):
    """Point init_weights at the GM checkpoint and lower LRs for fine-tuning."""
    import glob as _glob
    import yaml as _yaml

    cfgs = _glob.glob(
        os.path.join(project_dir, 'dlc-models', 'iteration-0', '*', 'train', 'pose_cfg.yaml')
    )
    if not cfgs:
        raise click.ClickException(
            "Could not find pose_cfg.yaml — create_training_dataset may have failed."
        )
    pose_cfg_path = cfgs[0]
    with open(pose_cfg_path, 'r') as _f:
        cfg = _yaml.safe_load(_f)

    cfg['init_weights'] = gm_snapshot_path
    cfg['multi_step'] = [[0.0005, 10000], [0.0001, 100000], [0.00005, 200000]]

    with open(pose_cfg_path, 'w') as _f:
        _yaml.dump(cfg, _f, default_flow_style=False)

    click.secho(f"→ fine-tuning from GM checkpoint: {gm_snapshot_path}", fg="green")



def analyze_and_ellipse(experiment, video_paths, config_path, plot_flag=False,
                       filter_flag=False, filter_params=None,
                       smooth_flag=False, smoothing_method='auto', smoothing_params=None,
                       make_labeled_video=False, gputouse=None):
    """Common: analyze video, fit ellipse, optionally filter and smooth, save CSV."""
    if filter_params is None:
        filter_params = {}
    if smoothing_params is None:
        smoothing_params = {}

    click.echo("→ running analysis…")
    patch_dlc_inference()
    deeplabcut.analyze_videos(config_path, video_paths, save_as_csv=True,
                              allow_growth=True, gputouse=gputouse, batchsize=64)

    if make_labeled_video:
        click.echo("→ creating annotated video…")
        deeplabcut.create_labeled_video(config_path, video_paths, save_frames=False)
    else:
        click.echo("→ skipping annotated video.")

    click.echo("→ analysis done.")
    
    for video_path in video_paths:
        prefix = os.path.splitext(os.path.basename(video_path))[0]
        viddir = os.path.dirname(video_path)

        # 1) find the exact DLC csv for this video
        pattern = f"{prefix}*DLC*.csv"
        matches = fnmatch.filter(os.listdir(viddir), pattern)
        if not matches:
            raise FileNotFoundError(f"No DLC CSV matching `{pattern}` in {viddir}")
        if len(matches) > 1:
            click.echo(f"⚠️  Warning: multiple matches for `{pattern}`; using first one")
        csv_file = matches[0]
        df = pd.read_csv(os.path.join(viddir, csv_file), low_memory=False)

        t0 = time.time()
        ell_df = ellipse_fitting(df)
        click.echo(f"→ ellipse fitting took {time.time()-t0:.1f}s")

        # compute diameter
        y1 = df.iloc[2:, -5].astype(float)
        y0 = df.iloc[2:, -11].astype(float)
        x1 = df.iloc[2:, -6].astype(float)
        x0 = df.iloc[2:, -12].astype(float)
        euc = ((y1-y0)**2 + (x1-x0)**2).pow(0.5)
        euc.index = ell_df.index
        ell_df['Eye_Diameter'] = euc

        # Keep track of processing steps
        processing_steps = []
        current_df = ell_df.copy()
        
        # Apply filtering first if requested
        if filter_flag:
            click.echo(f"→ applying rate-of-change filtering...")
            t1 = time.time()
            n_before = len(current_df)
            current_df = filter_by_rate_of_change(current_df, **filter_params)
            n_after = len(current_df)
            n_removed = n_before - n_after
            click.echo(f"→ filtering took {time.time()-t1:.1f}s, removed {n_removed} outliers ({n_removed/n_before*100:.1f}%)")
            processing_steps.append("filtered")
        
        # Apply smoothing after filtering if requested
        if smooth_flag:
            click.echo(f"→ applying {smoothing_method} smoothing...")
            t2 = time.time()
            current_df = smooth_pupil_data(
                current_df, 
                method=smoothing_method,
                save_original=False,  # Don't create duplicate columns
                **smoothing_params
            )
            click.echo(f"→ smoothing took {time.time()-t2:.1f}s")
            processing_steps.append("smoothed")
        
        # Save files based on processing applied
        if processing_steps:
            # Save raw data
            outpath_raw = os.path.join(viddir, f"PupilEye_{experiment}_{prefix}_raw.csv")
            ell_df.to_csv(outpath_raw, index=False)
            click.secho(f"→ raw ellipse CSV saved: {outpath_raw}", fg="yellow")
            
            # Save processed data as main output
            processing_suffix = "_".join(processing_steps)
            outpath = os.path.join(viddir, f"PupilEye_{experiment}_{prefix}.csv")
            current_df.to_csv(outpath, index=False)
            click.secho(f"→ {processing_suffix} ellipse CSV saved: {outpath}", fg="green")
            
            # For plotting
            plot_df = current_df
            plot_suffix = f"_{processing_suffix}"
            raw_df = ell_df
        else:
            # Save raw data as main output
            outpath = os.path.join(viddir, f"PupilEye_{experiment}_{prefix}.csv")
            ell_df.to_csv(outpath, index=False)
            click.secho(f"→ ellipse CSV saved: {outpath}", fg="green")
            
            plot_df = ell_df
            plot_suffix = ""
            raw_df = None
        
        if plot_flag:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Determine which pupil measurement to plot (prioritize Largest_Radius)
            pupil_column = 'Largest_Radius' if 'Largest_Radius' in plot_df.columns else 'Eye_Diameter'
            pupil_label = 'Pupil Radius (pixels)' if pupil_column == 'Largest_Radius' else 'Eye Diameter (pixels)'
            
            if smooth_flag and raw_df is not None:
                # Plot both original and smoothed data for comparison
                ax.plot(raw_df['Time_Frames'], raw_df[pupil_column], 
                       'o-', alpha=0.3, markersize=2, label='Raw data', color='lightblue')
                ax.plot(plot_df['Time_Frames'], plot_df[pupil_column], 
                       'r-', linewidth=2, label=f'Smoothed ({smoothing_method})')
                ax.legend()
            else:
                ax.plot(plot_df['Time_Frames'], plot_df[pupil_column], 'b-', linewidth=1)
                
            ax.set_xlabel('Frame')
            ax.set_ylabel(pupil_label)
            ax.set_title(f'{prefix}: Pupil Size over Time{plot_suffix}')
            ax.grid(True, alpha=0.3)

            out_png = os.path.join(viddir, f"PupilEye_{experiment}_{prefix}{plot_suffix}.png")
            fig.savefig(out_png, dpi=150, bbox_inches='tight')
            plt.close(fig)
            click.secho(f"→ pupil size plot saved: {out_png}", fg="cyan")

def _confirm(message, default=True):
    """click.confirm with a consistent lowercase [y/n] suffix regardless of default."""
    return click.confirm(message, default=default, show_default=False,
                         prompt_suffix=" [y/n]: ")


@click.command()
def main():
    click.clear()
    click.secho(pyfiglet.figlet_format("Pupil-DLC", font="slant"), fg="cyan")

    # choose path
    mode = click.prompt(
        "Model? [IM=Individual, GM=General, RT=ReTraining, FT=FineTuning]",
        type=click.Choice(["IM", "GM", "RT", "FT"]), default="GM"
    )
    default_dir = os.getcwd()
    working_dir = click.prompt(f"Working Directory [default: {default_dir}]",
                               type=str, default=default_dir, show_default=False)
    if sys.platform == "win32" and len(working_dir) > 80:
        click.secho(
            f"WARNING: Working directory path is {len(working_dir)} chars long. "
            "On Windows (MAX_PATH=260), DLC checkpoint paths can exceed this limit "
            "and cause a crash during training. Consider using a shorter path like D:\\FT_Work.",
            fg="yellow"
        )
    experiment = click.prompt("Experiment name", type=str)
    video_path = click.prompt(
        "Full path to your video file or folder", 
        type=click.Path(exists=True, file_okay=True, dir_okay=True)
    )

    plot_flag = _confirm(
        "Generate pupil-diameter-over-time plots (saved as PNG)?",
        default=False)

    make_labeled_video = _confirm(
        "Create annotated video (DeepLabCut labeled video)?",
        default=False
    )

# New filtering options
    filter_flag = _confirm(
        "Apply rate-of-change filtering to remove outliers/spikes?",
        default=True
    )
    
    filter_params = {}
    if filter_flag:
        lower_perc = click.prompt(
            "Lower percentile threshold for filtering", 
            type=float, default=5.0
        )
        upper_perc = click.prompt(
            "Upper percentile threshold for filtering", 
            type=float, default=95.0
        )
        filter_params = {'lower_perc': lower_perc, 'upper_perc': upper_perc}
    else:
        filter_params = {}

    # Smoothing options
    smooth_flag = _confirm(
        "Apply smoothing to pupil measurements?",
        default=True
    )
    
    smoothing_method = 'auto'  # default
    smoothing_params = {}
    if smooth_flag:
        smoothing_method = click.prompt(
            "Smoothing method",
            type=click.Choice(['auto', 'moving_average', 'gaussian', 'savgol', 'butterworth']),
            default='auto'
        )
        
        # Method-specific parameters
        if smoothing_method == 'moving_average':
            window_size = click.prompt("Window size for moving average", type=int, default=5)
            smoothing_params['window_size'] = window_size
        elif smoothing_method == 'gaussian':
            sigma = click.prompt("Sigma for Gaussian smoothing", type=float, default=1.5)
            smoothing_params['sigma'] = sigma
        elif smoothing_method == 'savgol':
            window_length = click.prompt("Window length (odd number)", type=int, default=11)
            polyorder = click.prompt("Polynomial order", type=int, default=3)
            smoothing_params['window_length'] = window_length
            smoothing_params['polyorder'] = polyorder
        elif smoothing_method == 'butterworth':
            cutoff_freq = click.prompt("Cutoff frequency (0-1)", type=float, default=0.1)
            smoothing_params['cutoff_freq'] = cutoff_freq
    else:
        smoothing_params = {}

    # collect all .avi/.mp4 files if a folder was given
    if os.path.isdir(video_path):
        video_paths = [
            os.path.join(video_path, f)
            for f in os.listdir(video_path)
            if f.lower().endswith(('.avi','.mp4'))
        ]
        if not video_paths:
            raise click.ClickException(f"No .avi or .mp4 files found in {video_path}")
    else:
        video_paths = [video_path]

    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
    default_config_path = os.path.normpath(os.path.join(repo_root, 'GM_Model', 'config.yaml'))

    if mode == "IM":
        gpu_number = click.prompt(
            "Which GPU to use?",
            type=click.IntRange(min=0),
            default=0,
            show_default=True
        )
        config_path = deeplabcut.create_new_project(
            experiment, "You", [video_path],
            working_directory=working_dir,
            copy_videos=False, multianimal=False
        )
        click.echo(f"→ project created: {config_path}")
        replace_yaml_section(config_path)

        while True:
            deeplabcut.extract_frames(config_path, mode="manual")
            deeplabcut.label_frames(config_path)
            if _confirm("Proceed to training?", default=True):
                break

        deeplabcut.check_labels(config_path, visualizeindividuals=True)
        deeplabcut.create_training_dataset(config_path, augmenter_type='imgaug')
        deeplabcut.train_network(
            config_path, shuffle=1, trainingsetindex=0,
            gputouse=gpu_number, max_snapshots_to_keep=5,
            autotune=False, displayiters=100,
            saveiters=15000, maxiters=500000, allow_growth=True
        )
        deeplabcut.evaluate_network(config_path, Shuffles=[1], plotting=False)

    elif mode == "GM":
        config_path = click.prompt(
            f"Full path to your config file [default: {default_config_path}]",
            default=default_config_path,
            type=click.Path(exists=True, dir_okay=False),
            show_default=False
        )
        click.echo(f"→ using config: {config_path}")
        _patch_config_project_path(config_path)

        gpu_number = click.prompt(
            "Which GPU to use? (0 = display GPU, 1 = non-display GPU — prefer 1 for speed)",
            type=click.IntRange(min=0),
            default=1,
            show_default=True
        )

    elif mode == "RT":
        gpu_number = click.prompt(
            "Which GPU to use? (0 = display GPU, 1 = non-display GPU — prefer 1 for speed)",
            type=click.IntRange(min=0),
            default=1,
            show_default=True
        )

        # Create a brand-new DLC project — GM_Model is never modified
        config_path = deeplabcut.create_new_project(
            experiment, "You", video_paths,
            working_directory=working_dir,
            copy_videos=False, multianimal=False
        )
        click.echo(f"→ RT project created: {config_path}")

        replace_yaml_section(config_path)

        project_dir = os.path.normpath(os.path.dirname(config_path))
        _restore_moved_videos(video_paths, project_dir)

        # Read the scorer DLC assigned (derived from the experimenter name prompt)
        import yaml as _yaml
        with open(config_path, 'r') as _f:
            _scorer = _yaml.safe_load(_f).get('scorer', 'You')

        # Download GM data to global cache (once ever) — do this before labeling so
        # the user isn't waiting mid-session. Does NOT touch video_sets yet.
        _ensure_gm_cache()

        # Extract frames from user's video for labeling.
        # video_sets only contains the user's real video at this point.
        frame_mode = click.prompt(
            "Frame extraction mode",
            type=click.Choice(['automatic', 'manual']),
            default='automatic'
        )
        click.secho("→ extracting frames from your video for labeling…", fg="yellow")
        deeplabcut.extract_frames(config_path, mode=frame_mode,
                                  algo='uniform', userfeedback=False)

        click.secho(
            "→ label the extracted frames in the GUI, then close it to continue.",
            fg="yellow"
        )
        _patch_dlc_labeling_toolbox()
        deeplabcut.label_frames(config_path)

        if not _confirm("Labeling complete? Proceed to retraining?", default=True):
            click.echo("Aborted. Re-run RT mode when labeling is done.")
            return

        _result = _verify_and_relabel(config_path, project_dir, video_paths, _scorer)
        if _result is None:
            return

        # Now link GM frames into labeled-data/ and register them in video_sets —
        # after extract_frames so DLC never tries to open the placeholder video paths.
        _link_gm_data_to_project(project_dir, _scorer)

        deeplabcut.check_labels(config_path)
        deeplabcut.create_training_dataset(config_path, augmenter_type='imgaug')

        # Verify the training dataset is non-empty before starting — an empty
        # dataset causes DLC's data-loader thread to crash with ValueError and
        # the main training loop hangs indefinitely.
        import glob as _glob
        _labeled_imgs = _glob.glob(
            os.path.join(os.path.dirname(config_path), "labeled-data", "**", "img*.png"),
            recursive=True
        )
        if not _labeled_imgs:
            raise click.ClickException(
                "No labeled images found in labeled-data/. "
                "Make sure you saved your labels in the GUI and that the GM data downloaded correctly."
            )

        max_iters = click.prompt(
            "Max training iterations",
            type=int, default=200000
        )
        deeplabcut.train_network(
            config_path, shuffle=1, trainingsetindex=0,
            gputouse=gpu_number, max_snapshots_to_keep=5,
            autotune=False, displayiters=100,
            saveiters=15000, maxiters=max_iters, allow_growth=True
        )
        deeplabcut.evaluate_network(config_path, Shuffles=[1], plotting=False)

        if not _confirm("Run inference on your video with the retrained model?", default=True):
            return

    elif mode == "FT":
        gpu_number = click.prompt(
            "Which GPU to use? (0 = display GPU, 1 = non-display GPU — prefer 1 for speed)",
            type=click.IntRange(min=0),
            default=1,
            show_default=True
        )

        # Allow skipping straight to inference on an already-trained FT project.
        _ft_inference_only = _confirm(
            "Skip training — run inference on an existing FT project?",
            default=False
        )

        if _ft_inference_only:
            config_path = click.prompt(
                "Path to the existing FT project config.yaml",
                type=click.Path(exists=True, dir_okay=False)
            )
            project_dir = os.path.normpath(os.path.dirname(config_path))
            # Videos live in the project's videos/ folder (DLC may have moved them there)
            _videos_dir = os.path.join(project_dir, "videos")
            video_paths = [
                os.path.join(_videos_dir, f)
                for f in os.listdir(_videos_dir)
                if f.lower().endswith(('.avi', '.mp4'))
            ] if os.path.isdir(_videos_dir) else video_paths

        else:
            config_path = deeplabcut.create_new_project(
                experiment, "You", video_paths,
                working_directory=working_dir,
                copy_videos=False, multianimal=False
            )
            click.echo(f"→ FT project created: {config_path}")

            replace_yaml_section(config_path)

            project_dir = os.path.normpath(os.path.dirname(config_path))

            _restore_moved_videos(video_paths, project_dir)

            import yaml as _yaml
            with open(config_path, 'r') as _f:
                _scorer = _yaml.safe_load(_f).get('scorer', 'You')

            frame_mode = click.prompt(
                "Frame extraction mode",
                type=click.Choice(['automatic', 'manual']),
                default='automatic'
            )
            click.secho("→ extracting frames from your video for labeling…", fg="yellow")
            deeplabcut.extract_frames(config_path, mode=frame_mode,
                                      algo='uniform', userfeedback=False)

            click.secho(
                "→ label the extracted frames in the GUI, then close it to continue.",
                fg="yellow"
            )
            _patch_dlc_labeling_toolbox()
            deeplabcut.label_frames(config_path)

            if not _confirm("Labeling complete? Proceed to fine-tuning?", default=True):
                click.echo("Aborted. Re-run FT mode when labeling is done.")
                return

            _result = _verify_and_relabel(config_path, project_dir, video_paths, _scorer)
            if _result is None:
                return

            deeplabcut.check_labels(config_path)
            deeplabcut.create_training_dataset(config_path, augmenter_type='imgaug')

            gm_snapshot = _find_gm_snapshot(repo_root)
            _patch_pose_cfg_for_finetuning(project_dir, gm_snapshot)

            max_iters = click.prompt(
                "Max fine-tuning iterations",
                type=int, default=50000
            )
            deeplabcut.train_network(
                config_path, shuffle=1, trainingsetindex=0,
                gputouse=gpu_number, max_snapshots_to_keep=5,
                autotune=False, displayiters=100,
                saveiters=15000, maxiters=max_iters, allow_growth=True
            )
            deeplabcut.evaluate_network(config_path, Shuffles=[1], plotting=False)

        if not _confirm("Run inference on your video with the fine-tuned model?", default=True):
            return

    # IM, GM, RT, and FT (if inference confirmed) all converge here
    analyze_and_ellipse(
        experiment=experiment,
        video_paths=video_paths,
        config_path=config_path,
        plot_flag=plot_flag,
        filter_flag=filter_flag,
        filter_params=filter_params,
        smooth_flag=smooth_flag,
        smoothing_method=smoothing_method,
        smoothing_params=smoothing_params,
        make_labeled_video=make_labeled_video,
        gputouse=gpu_number,
    )

if __name__ == '__main__':
    main()
