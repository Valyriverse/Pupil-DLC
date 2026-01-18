###################### Pupil-DLC Pipeline#########################

#!/usr/bin/env python
import os
os.environ["MPLBACKEND"] = "Agg"
#import matplotlib
#matplotlib.use('Agg')
import sys
import shutil
import click
import pyfiglet
import pandas as pd
import deeplabcut
import fnmatch
import time

##### Pupil reinstallment should be in the main folder where the setup.py is and do pip install -e .
from .smoothing_module import smooth_pupil_data, filter_by_rate_of_change
from .ellipse import ellipse_fitting
from .yaml_section import replace_yaml_section

def analyze_and_ellipse(experiment, video_paths, config_path, plot_flag=False, 
                       filter_flag=False, filter_params=None,
                       smooth_flag=False, smoothing_method='auto', smoothing_params=None,
                       make_labeled_video=False):
    """Common: analyze video, fit ellipse, optionally filter and smooth, save CSV."""
    if filter_params is None:
        filter_params = {}
    if smoothing_params is None:
        smoothing_params = {}
        
    click.echo("→ running analysis…")
    deeplabcut.analyze_videos(config_path, video_paths, save_as_csv=True)

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

@click.command()
def main():
    click.clear()
    click.secho(pyfiglet.figlet_format("Pupil-DLC", font="slant"), fg="cyan")

    # choose path
    mode = click.prompt(
        "Model? [IM=Individual, GM=General]", 
        type=click.Choice(["IM","GM"]), default="Default: GM"
    )
    default_dir = os.getcwd()
    working_dir = click.prompt(f"Working Directory [default: {default_dir}]", 
                               type=str, default=default_dir, show_default=False)
    experiment = click.prompt("Experiment name", type=str)
    video_path = click.prompt(
        "Full path to your video file or folder", 
        type=click.Path(exists=True, file_okay=True, dir_okay=True)
    )

    plot_flag = click.confirm(
    "Generate pupil-diameter-over-time plots (saved as PNG)?",
    default=False)

    make_labeled_video = click.confirm(
        "Create annotated video (DeepLabCut labeled video)?",
        default=False
    )


# New filtering options
    filter_flag = click.confirm(
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
    smooth_flag = click.confirm(
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

    if mode == "IM":
        gpu_number = click.prompt(
        "Which GPU to use?",
        type=click.IntRange(min=0),
        default=0,
        show_default=True
        )
        # create a fresh project
        config_path = deeplabcut.create_new_project(
            experiment, "You", [video_path],
            working_directory=working_dir,
            copy_videos=False, multianimal=False
        )
        click.echo(f"→ project created: {config_path}")
        replace_yaml_section(config_path)

        # loop: label → ask to proceed
        while True:
            deeplabcut.extract_frames(config_path, mode="manual")
            deeplabcut.label_frames(config_path)
            if click.confirm("Proceed to training?"):
                break

        deeplabcut.check_labels(config_path, visualizeindividuals=True)
        deeplabcut.create_training_dataset(config_path, augmenter_type='imgaug')
        deeplabcut.train_network(
            config_path, shuffle=1, trainingsetindex=0,
            gputouse=gpu_number, max_snapshots_to_keep=5,
            autotune=False, displayiters=100,
            saveiters=15000, maxiters=500000, allow_growth=True
        )
        deeplabcut.evaluate_network(config_path, Shuffles=[1], plotting=True)

    else:  # mode == "GM"
        # user already has a config.yaml
        #config_path = click.prompt(
        #    "Full path to your existing config file",
        #    type=click.Path(exists=True, dir_okay=False)
        #)

        # resolve path to the default general model config.yaml inside your repo
        repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
        default_config_path = os.path.join(repo_root, 'GM_gitub', 'config.yaml')
        default_config_path = os.path.normpath(default_config_path)

        config_path = click.prompt(
            f"Full path to your config file [default: {default_config_path}]",
            default=default_config_path,
            type=click.Path(exists=True, dir_okay=False),
            show_default=False
        )
        click.echo(f"→ using config: {config_path}")
        # (any GM-specific prep you already have…)

    # both IM & GM converge here:
    if plot_flag:
        import matplotlib.pyplot as plt  # add at top of file
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
        make_labeled_video=make_labeled_video
    )

if __name__ == '__main__':
    main()
