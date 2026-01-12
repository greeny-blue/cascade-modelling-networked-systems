import numpy as np
import pygame
import random
import time
import sys
import argparse
import os

# ------------------------------------------------------------
# SELF-ORGANISED CRITICAL FOREST FIRE (SOC VERSION, ANIMATED)
# ------------------------------------------------------------
# States:
#   1 = tree
#   2 = burning tree
#   3 = soil / empty
#
# Modes:
#   "DRIVE": slow tree growth + rare lightning
#   "FIRE":  active burning until all fires are out
#
# This script can run in:
#   - interactive mode (prompting for parameters)
#   - or CLI mode (parameters from command-line)
#
# It logs fire sizes and (optionally) density over time to text files,
# and can also store "pre-burn" forest states (trees vs soil) in a .npz
# for offline analysis (e.g. cluster-based correlation length).
# ------------------------------------------------------------

# ------------------------ defaults -------------------------- #

GRID_SIZE_DEFAULT = 256            # cells per side
INIT_TREE_DEFAULT = 0.01           # initial fraction of trees
P_GROW_DEFAULT = 0.01              # probability of tree growth per cell per DRIVE step
P_LIGHTNING_DEFAULT = 1e-5         # probability of lightning per cell per DRIVE step
STEPS_PER_FRAME_DEFAULT = 1000      # how many DRIVE steps per video frame
CELL_SIZE_DEFAULT = 4              # pixel size of each cell
FPS_DEFAULT = 60                   # frames per second for animation

LOG_FILENAME_DEFAULT = "socsim_output_fire_sizes.txt"

# Extra defaults
MAX_RUNS_DEFAULT = -1               # -1 or <=0 means "infinite"
FIRE_STEPS_PER_FRAME_DEFAULT = 3    # how many fire CA steps per video frame
NO_GRAPHICS_DEFAULT = True          # default: show graphics
LIMIT_BY_DEFAULT = "fires"          # "fires" or "frames"
TRACK_DENSITY_DEFAULT = True
DENSITY_LOG_INTERVAL_DEFAULT = 1000  # in frames (or DRIVE updates)
DENSITY_LOG_FILENAME_DEFAULT = "socsim_output_density_timeseries.txt"
PREBURN_STATES_FILENAME_DEFAULT = "socsim_output_preburn_states.npz"
AUTO_SCREENSHOTS_DEFAULT = False
AUTO_SCREENSHOT_DIR_DEFAULT = "socsim_automatic_screenshots"


# ------------- simple prompt helpers (for --interactive) ------------- #

def _prompt_int(message, default, min_val=None, max_val=None):
    while True:
        raw = input(f"{message} [{default}]: ").strip()
        if raw == "":
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                print("Please enter an integer.")
                continue

        if min_val is not None and value < min_val:
            print(f"Value must be >= {min_val}.")
            continue
        if max_val is not None and value > max_val:
            print(f"Value must be <= {max_val}.")
            continue
        return value


def _prompt_float(message, default, min_val=None, max_val=None):
    while True:
        raw = input(f"{message} [{default}]: ").strip()
        if raw == "":
            value = default
        else:
            try:
                value = float(raw)
            except ValueError:
                print("Please enter a number.")
                continue

        if min_val is not None and value < min_val:
            print(f"Value must be >= {min_val}.")
            continue
        if max_val is not None and value > max_val:
            print(f"Value must be <= {max_val}.")
            continue
        return value


def _prompt_seed(message, default):
    while True:
        raw = input(f"{message} [{default} or 'none']: ").strip().lower()
        if raw == "":
            return default
        if raw == "none":
            return None
        try:
            return int(raw)
        except ValueError:
            print("Please enter an integer or 'none'.")


def _prompt_bool(message, default):
    default_str = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{message} [{default_str}]: ").strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Please answer y or n.")


# ------------------------- CLI argument parsing ---------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Self-organised critical forest-fire model (SOC)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--grid-size", type=int, default=GRID_SIZE_DEFAULT,
                        help="Grid size (cells per side)")
    parser.add_argument("--prop-tree-init", type=float, default=INIT_TREE_DEFAULT,
                        help="Initial tree cover proportion")
    parser.add_argument("--p-grow", type=float, default=P_GROW_DEFAULT,
                        help="Tree growth probability per cell per DRIVE step")
    parser.add_argument("--p-lightning", type=float, default=P_LIGHTNING_DEFAULT,
                        help="Lightning probability per cell per DRIVE step")
    parser.add_argument("--steps-per-frame", type=int, default=STEPS_PER_FRAME_DEFAULT,
                        help="Number of DRIVE steps per frame")
    parser.add_argument("--cell-size", type=int, default=CELL_SIZE_DEFAULT,
                        help="Cell size in pixels")
    parser.add_argument("--fps", type=int, default=FPS_DEFAULT,
                        help="Frames per second")
    parser.add_argument("--max-runs", type=int, default=MAX_RUNS_DEFAULT,
                        help="If > 0, stop after this many fires/frames (see --limit-by)")
    parser.add_argument("--fire-steps-per-frame", type=int, default=FIRE_STEPS_PER_FRAME_DEFAULT,
                        help="Number of CA steps per frame while FIRE is burning")
    parser.add_argument("--no-graphics", action="store_true", default=NO_GRAPHICS_DEFAULT,
                        help="Run without graphics (faster)")
    parser.add_argument("--limit-by", choices=["fires", "frames"], default=LIMIT_BY_DEFAULT,
                        help="If max-runs > 0, choose whether to limit by number of fires or number of frames")
    parser.add_argument("--track-density", action="store_true", default=TRACK_DENSITY_DEFAULT,
                        help="Track and log tree density over time")
    parser.add_argument("--density-log-interval", type=int, default=DENSITY_LOG_INTERVAL_DEFAULT,
                        help="Interval in frames for recording density")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: None, i.e. random)")
    parser.add_argument("--auto-screenshots", action="store_true", default=AUTO_SCREENSHOTS_DEFAULT,
                        help="Automatically save screenshots of all fire steps (graphics mode only)")
    parser.add_argument("--screenshot-dir", type=str, default=AUTO_SCREENSHOT_DIR_DEFAULT,
                        help="Directory for automatic fire screenshots")

    parser.add_argument("--interactive", action="store_true",
                        help="Prompt for parameters interactively")

    return parser.parse_args()


# --------------------------- param handling -------------------------- #

def get_user_params():
    """
    Returns a dictionary of parameters, either from CLI or interactive prompts.
    """
    args = parse_args()

    if not args.interactive:
        params = {
            "grid_size": args.grid_size,
            "prop_tree_init": args.prop_tree_init,
            "p_grow": args.p_grow,
            "p_lightning": args.p_lightning,
            "steps_per_frame": args.steps_per_frame,
            "cell_size": args.cell_size,
            "fps": args.fps,
            "max_runs": args.max_runs,
            "fire_steps_per_frame": args.fire_steps_per_frame,
            "no_graphics": args.no_graphics,
            "limit_by_fires": (args.limit_by == "fires"),
            "track_density": args.track_density,
            "density_log_interval": args.density_log_interval,
            "seed": args.seed,
            "auto_screenshots": args.auto_screenshots,
            "screenshot_dir": args.screenshot_dir,
        }
        return params

    # Interactive mode
    print("Interactive mode: configure the SOC forest-fire simulation.\n")

    grid_size = _prompt_int("Grid size (cells per side)", GRID_SIZE_DEFAULT, min_val=10)
    prop_tree_init = _prompt_float("Initial tree cover proportion", INIT_TREE_DEFAULT, min_val=0.0, max_val=1.0)
    p_grow = _prompt_float("Tree growth probability per cell per DRIVE step", P_GROW_DEFAULT, min_val=0.0)
    p_lightning = _prompt_float("Lightning probability per cell per DRIVE step", P_LIGHTNING_DEFAULT, min_val=0.0)
    steps_per_frame = _prompt_int("Number of DRIVE steps per frame", STEPS_PER_FRAME_DEFAULT, min_val=1)
    cell_size = _prompt_int("Cell size in pixels", CELL_SIZE_DEFAULT, min_val=1)
    fps = _prompt_int("Frames per second", FPS_DEFAULT, min_val=1)
    max_runs = _prompt_int("Max fires/frames (0 or negative = infinite)", MAX_RUNS_DEFAULT)
    fire_steps_per_frame = _prompt_int("Fire CA steps per frame", FIRE_STEPS_PER_FRAME_DEFAULT, min_val=1)
    no_graphics = _prompt_bool("Disable graphics for speed?", NO_GRAPHICS_DEFAULT)
    limit_by_fires = _prompt_bool("Limit by fires? (otherwise by frames)", LIMIT_BY_DEFAULT == "fires")
    track_density = _prompt_bool("Track and log tree density over time?", TRACK_DENSITY_DEFAULT)
    density_log_interval = _prompt_int("Density log interval (frames)", DENSITY_LOG_INTERVAL_DEFAULT, min_val=1)
    auto_screenshots = _prompt_bool("Automatically save screenshots of all fire steps? (graphics mode only)", AUTO_SCREENSHOTS_DEFAULT)
    seed = _prompt_seed("Random seed (int or 'none')", None)

    params = {
        "grid_size": grid_size,
        "prop_tree_init": prop_tree_init,
        "p_grow": p_grow,
        "p_lightning": p_lightning,
        "steps_per_frame": steps_per_frame,
        "cell_size": cell_size,
        "fps": fps,
        "max_runs": max_runs,
        "fire_steps_per_frame": fire_steps_per_frame,
        "no_graphics": no_graphics,
        "limit_by_fires": limit_by_fires,
        "track_density": track_density,
        "density_log_interval": density_log_interval,
        "seed": seed,
        "auto_screenshots": auto_screenshots,
        "screenshot_dir": AUTO_SCREENSHOT_DIR_DEFAULT,
    }
    return params


# ---------------------- neighbours & drawing ------------------------ #

def get_neighbours(i, j, n_rows, n_cols):
    """
    Von Neumann neighbourhood (up, down, left, right) with open boundaries.
    """
    neighbours = []
    if i > 0:
        neighbours.append((i - 1, j))
    if i < n_rows - 1:
        neighbours.append((i + 1, j))
    if j > 0:
        neighbours.append((i, j - 1))
    if j < n_cols - 1:
        neighbours.append((i, j + 1))
    return neighbours


def draw_cell(screen, state_value, cell_size, i, j):
    """
    Draw a single cell at (i, j) given its state_value.
    """
    if state_value == 1:
        color = (34, 139, 34)   # tree = green
    elif state_value == 2:
        color = (255, 218, 0)  # burning = orange-red
    else:
        color = (20, 20, 20)  # soil = dark grey

    rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
    pygame.draw.rect(screen, color, rect)


def draw_state(screen, state, cell_size):
    """
    Draw the entire forest state to the screen.
    """
    n_rows, n_cols = state.shape
    for i in range(n_rows):
        for j in range(n_cols):
            draw_cell(screen, state[i, j], cell_size, i, j)


# -------------------------- logging helpers ------------------------- #

def _format_seconds(seconds):
    # Simple formatting: H:MM:SS
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    else:
        return f"{m:d}:{s:02d}"


def update_progress(completed, total, start_time, last_progress_time, label="Progress"):
    """
    Simple text-based progress bar for CLI.
    """
    now = time.time()
    if now - last_progress_time < 0.5:
        return last_progress_time

    if total <= 0:
        sys.stdout.write(f"\r{label}: {completed}")
        sys.stdout.flush()
        return now

    frac = completed / total
    bar_width = 30
    filled = int(frac * bar_width)
    bar = "#" * filled + "-" * (bar_width - filled)

    elapsed = now - start_time
    if completed > 0:
        rate = completed / elapsed
        remaining = (total - completed) / rate if rate > 0 else 0.0
    else:
        remaining = 0.0

    eta_str = _format_seconds(remaining)
    pct = int(frac * 100)

    msg = f"\r{label}: [{bar}] {completed}/{total} ({pct}%) ETA {eta_str}"
    sys.stdout.write(msg)
    sys.stdout.flush()

    if completed >= total:
        sys.stdout.write("\n")

    return now


# -------------------------- main simulation -------------------------- #

def main():
    params = get_user_params()

    grid_size = params["grid_size"]
    prop_tree_init = params["prop_tree_init"]
    p_grow = params["p_grow"]
    p_lightning = params["p_lightning"]
    steps_per_frame = params["steps_per_frame"]
    cell_size = params["cell_size"]
    fps = params["fps"]
    limit_by_fires = params["limit_by_fires"]
    max_runs = params["max_runs"]
    fire_steps_per_frame = params["fire_steps_per_frame"]
    no_graphics = params["no_graphics"]
    seed = params["seed"]
    track_density = params["track_density"]
    density_log_interval = params["density_log_interval"]
    auto_screenshots = params.get("auto_screenshots", AUTO_SCREENSHOTS_DEFAULT)
    screenshot_dir = params.get("screenshot_dir", AUTO_SCREENSHOT_DIR_DEFAULT)

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Initial tree configuration
    state = rng.choice(
        [1, 3],
        size=(grid_size, grid_size),
        p=[prop_tree_init, 1 - prop_tree_init],
    ).astype(np.int8)
    n_rows, n_cols = state.shape

    # Set up pygame if graphics are enabled
    if not no_graphics:
        pygame.init()
        width = grid_size * cell_size
        height = grid_size * cell_size
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Self-organised critical forest fire (animated)")
        clock = pygame.time.Clock()
        # Prepare directory for automatic screenshots if requested
        if auto_screenshots:
            os.makedirs(screenshot_dir, exist_ok=True)
    else:
        screen = None
        clock = None

    paused = False
    running = True

    # Screenshot bookkeeping
    manual_screenshot_index = 0
    current_fire_index = 0  # increments for each new fire when auto_screenshots is True

    mode = "DRIVE"            # or "FIRE"
    current_fire_size = 0     # counts trees burned in the current fire
    current_fire_duration = 0  # counts CA steps in the current fire
    frame_count = 0           # counts frames
    fire_count = 0            # counts completed fires
    fire_sizes = []           # store all completed fire sizes
    fire_durations = []       # store duration (CA steps) of each fire
    density_records = []      # list of (frame, rho)
    pre_fire_states = []      # store pre-burn forest states for offline analysis
    DENSITY_LOG_FILENAME = DENSITY_LOG_FILENAME_DEFAULT

    log_filename = LOG_FILENAME_DEFAULT

    # Progress tracking
    start_time = time.time()
    last_progress_time = start_time
    progress_label = "Fires" if limit_by_fires else "Frames"

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------
    while running:
        # Handle events only if we have graphics
        if not no_graphics:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key in (pygame.K_SPACE, pygame.K_p):
                        paused = not paused
                    elif event.key == pygame.K_r:
                        # Reset to empty soil, back to DRIVE mode
                        state[:] = 3
                        mode = "DRIVE"
                        current_fire_size = 0
                        current_fire_duration = 0
                    elif event.key == pygame.K_s:
                        # Manual screenshot
                        if screen is not None:
                            manual_screenshot_index += 1
                            timestamp_ms = int(time.time() * 1000)
                            filename = f"socsim_manual_{manual_screenshot_index:04d}_{timestamp_ms}.png"
                            pygame.image.save(screen, filename)
                            print(f"Saved manual screenshot to {filename}")

        if not paused:
            if mode == "DRIVE":
                # --------------------------------------------
                # SLOW DRIVE: growth + rare lightning
                # --------------------------------------------
                n_grow = int(steps_per_frame * p_grow)

                expected_lightning = steps_per_frame * p_lightning
                n_lightning = rng.poisson(expected_lightning) if expected_lightning > 0 else 0

                # Tree growth attempts
                for _ in range(n_grow):
                    i = rng.integers(0, n_rows)
                    j = rng.integers(0, n_cols)
                    if state[i, j] == 3:  # soil / empty
                        state[i, j] = 1

                # Lightning attempts
                for _ in range(n_lightning):
                    i = rng.integers(0, n_rows)
                    j = rng.integers(0, n_cols)
                    if state[i, j] == 1 and mode == "DRIVE":
                        # Record the pre-burn forest state (trees vs soil) for offline analysis
                        pre_fire_states.append((state == 1).astype(np.int8))
                        # Start a new fire: ignite this tree, switch to FIRE mode
                        state[i, j] = 2
                        mode = "FIRE"
                        current_fire_size = 0
                        current_fire_duration = 0
                        if auto_screenshots and not no_graphics:
                            current_fire_index += 1
                        break  # only one fire at a time

                if track_density and frame_count % density_log_interval == 0:
                    # rho = fraction of trees
                    rho = np.mean(state == 1)
                    density_records.append((frame_count, float(rho)))

            elif mode == "FIRE":
                # --------------------------------------------
                # FIRE MODE: animate a single fire
                # --------------------------------------------
                for _ in range(fire_steps_per_frame):
                    burning_mask = (state == 2)
                    burning_count = int(burning_mask.sum())

                    if burning_count == 0:
                        # Fire has burnt out – store size & duration, return to DRIVE mode
                        if current_fire_size > 0:
                            fire_sizes.append(current_fire_size)
                            fire_durations.append(current_fire_duration)
                            fire_count += 1
                        current_fire_size = 0
                        current_fire_duration = 0
                        mode = "DRIVE"
                        break
                    else:
                        # One CA step of the fire
                        current_fire_duration += 1
                        current_fire_size += burning_count

                        # Next state for fire spread
                        new_state = state.copy()

                        burning_indices = np.argwhere(burning_mask)
                        for i, j in burning_indices:
                            new_state[i, j] = 3  # burnt tree becomes soil
                            for ni, nj in get_neighbours(i, j, n_rows, n_cols):
                                if state[ni, nj] == 1:
                                    new_state[ni, nj] = 2  # ignite tree

                        state = new_state

            # Update frame count
            frame_count += 1

            # Check stopping condition if max_runs > 0
            if max_runs > 0:
                if limit_by_fires:
                    completed = fire_count
                else:
                    completed = frame_count

                if completed >= max_runs:
                    running = False

                # Progress bar
                completed_clamped = min(completed, max_runs)
                last_progress_time = update_progress(
                    completed=completed_clamped,
                    total=max_runs,
                    start_time=start_time,
                    last_progress_time=last_progress_time,
                    label=progress_label,
                )

        # ---------------- RENDERING ----------------
        if not no_graphics:
            screen.fill((0, 0, 0))
            draw_state(screen, state, cell_size)
            pygame.display.flip()

            # Automatic screenshots: capture every frame while a fire is burning
            if auto_screenshots and mode == "FIRE" and current_fire_index > 0:
                filename = os.path.join(
                    screenshot_dir,
                    f"fire_{current_fire_index:05d}_step_{current_fire_duration:05d}.png",
                )
                pygame.image.save(screen, filename)

            clock.tick(fps)

    # --------------- END OF MAIN LOOP -----------------

    # Clean up pygame
    if not no_graphics:
        pygame.quit()

    # Save fire sizes to file
    if fire_sizes:
        with open(LOG_FILENAME_DEFAULT, "w") as f:
            for size in fire_sizes:
                f.write(f"{size}\n")
        print(f"Saved {len(fire_sizes)} fire sizes to {LOG_FILENAME_DEFAULT}")

    # Save fire durations
    if fire_durations:
        durations_filename = LOG_FILENAME_DEFAULT.replace(".txt", "_durations.txt")
        with open(durations_filename, "w") as f:
            for d in fire_durations:
                f.write(f"{d}\n")
        print(f"Saved {len(fire_durations)} fire durations to {durations_filename}")

    # Save density records if tracked
    if track_density and density_records:
        with open(DENSITY_LOG_FILENAME, "w") as f:
            for frame, rho in density_records:
                f.write(f"{frame}\t{rho}\n")
        print(f"Saved density time series to {DENSITY_LOG_FILENAME}")

    # Save pre-burn forest states
    if pre_fire_states:
        arr = np.stack(pre_fire_states, axis=0)  # shape: (n_fires, n_rows, n_cols)
        np.savez_compressed(PREBURN_STATES_FILENAME_DEFAULT, preburn_states=arr)
        print(f"Saved {arr.shape[0]} pre-burn states to {PREBURN_STATES_FILENAME_DEFAULT}")


if __name__ == "__main__":
    main()

