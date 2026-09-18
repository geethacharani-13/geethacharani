
import numpy as np
import networkx as nx
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.optimize import linear_sum_assignment



NAME = "GEETHA"

N = 20                  # FIXED as required
P = 0.25                # Erdos-Renyi probability

GRAPH_SEED = 42
POSITION_SEED = 7

DT = 0.02

# Time used to form each letter
TIME_PER_LETTER = 3.0

# Pause after completing each letter
PAUSE_TIME = 0.8

# Coordinate system
CENTER = np.array([5.0, 5.0])

LETTER_WIDTH = 2.8
LETTER_HEIGHT = 4.0


seed = GRAPH_SEED

while True:

    G = nx.erdos_renyi_graph(
        N,
        P,
        seed=seed
    )

    if nx.is_connected(G):
        break

    seed += 1


print("=" * 65)
print("FORMATION CONTROL ON ERDOS-RENYI GRAPH")
print("=" * 65)

print("Name              :", NAME)
print("Number of agents  :", N)
print("Edge probability  :", P)
print("Graph seed        :", seed)
print("Number of edges   :", G.number_of_edges())
print("Connected         :", nx.is_connected(G))


L = nx.laplacian_matrix(G).toarray().astype(float)

rng = np.random.default_rng(
    POSITION_SEED
)

x_initial = rng.uniform(
    0,
    10,
    size=(N, 2)
)

LETTER_STROKES = {

    "A": [
        [(-1, -1), (0, 1), (1, -1)],
        [(-0.55, -0.05), (0.55, -0.05)]
    ],

    "E": [
        [(-0.8, 1), (-0.8, -1)],
        [(-0.8, 1), (0.8, 1)],
        [(-0.8, 0), (0.55, 0)],
        [(-0.8, -1), (0.8, -1)]
    ],

    "G": [
        [
            (0.8, 0.8),
            (0.35, 1),
            (-0.45, 1),
            (-0.8, 0.5),
            (-0.8, -0.5),
            (-0.35, -1),
            (0.45, -1),
            (0.8, -0.6),
            (0.8, -0.1),
            (0.1, -0.1)
        ]
    ],

    "H": [
        [(-0.8, -1), (-0.8, 1)],
        [(0.8, -1), (0.8, 1)],
        [(-0.8, 0), (0.8, 0)]
    ],

    "T": [
        [(-0.9, 1), (0.9, 1)],
        [(0, 1), (0, -1)]
    ],
}


def sample_polyline(points, n_points):

    points = np.asarray(
        points,
        dtype=float
    )

    if len(points) == 1:

        return np.repeat(
            points,
            n_points,
            axis=0
        )

    segments = np.diff(
        points,
        axis=0
    )

    lengths = np.linalg.norm(
        segments,
        axis=1
    )

    cumulative = np.concatenate(
        ([0.0], np.cumsum(lengths))
    )

    total_length = cumulative[-1]

    if total_length == 0:

        return np.repeat(
            points[:1],
            n_points,
            axis=0
        )

    distances = np.linspace(
        0,
        total_length,
        n_points
    )

    sampled = []

    for d in distances:

        idx = np.searchsorted(
            cumulative,
            d,
            side="right"
        ) - 1

        idx = min(
            idx,
            len(segments) - 1
        )

        segment_length = lengths[idx]

        if segment_length == 0:

            alpha = 0.0

        else:

            alpha = (
                d - cumulative[idx]
            ) / segment_length

        point = (
            points[idx]
            + alpha * segments[idx]
        )

        sampled.append(point)

    return np.array(sampled)


def generate_letter_points(
    letter,
    n_points=N
):
    

    letter = letter.upper()

    if letter not in LETTER_STROKES:

        raise ValueError(
            f"Letter {letter} is not supported."
        )

    strokes = LETTER_STROKES[letter]

    # Calculate length of every stroke
    stroke_lengths = []

    for stroke in strokes:

        stroke = np.asarray(
            stroke,
            dtype=float
        )

        diff = np.diff(
            stroke,
            axis=0
        )

        length = np.sum(
            np.linalg.norm(
                diff,
                axis=1
            )
        )

        stroke_lengths.append(
            length
        )

    stroke_lengths = np.array(
        stroke_lengths
    )

    total_length = np.sum(
        stroke_lengths)

    raw_counts = (
        stroke_lengths
        / total_length
        * n_points
    )

    counts = np.floor(
        raw_counts
    ).astype(int)

    # Every stroke gets at least one point
    counts = np.maximum(
        counts,
        1
    )

    # Correct total number of points
    while np.sum(counts) < n_points:

        remaining = (
            raw_counts - counts
        )

        idx = np.argmax(
            remaining
        )

        counts[idx] += 1

    while np.sum(counts) > n_points:

        valid = np.where(
            counts > 1
        )[0]

        if len(valid) == 0:
            break

        idx = valid[
            np.argmax(
                counts[valid]
            )
        ]

        counts[idx] -= 1

    # Sample each stroke

    all_points = []

    for stroke, count in zip(
        strokes,
        counts
    ):

        sampled = sample_polyline(
            stroke,
            count
        )

        all_points.append(
            sampled
        )

    points = np.vstack(
        all_points
    )
    # Scaling the letter
    

    min_xy = points.min(
        axis=0
    )

    max_xy = points.max(
        axis=0
    )

    center = (
        min_xy + max_xy
    ) / 2

    span = (
        max_xy - min_xy
    )

    span[span == 0] = 1

    # Keep aspect ratio
    scale = min(
        LETTER_WIDTH / span[0],
        LETTER_HEIGHT / span[1]
    )

    points = (
        points - center
    ) * scale

    points += CENTER

    return points

# 8. GENERATE TARGET FORMATION FOR EACH UNIQUE LETTER


unique_letters = list(
    dict.fromkeys(
        NAME.upper()
    )
)

letter_targets = {}

for letter in unique_letters:

    letter_targets[letter] = (
        generate_letter_points(
            letter,
            N
        )
    )

# ASSIGN AGENTS TO TARGET POINTS

def assign_targets(
    current_positions,
    target_positions
):
    

    difference = (
        current_positions[:, None, :]
        - target_positions[None, :, :]
    )

    cost = np.sum(
        difference ** 2,
        axis=2
    )

    rows, columns = (
        linear_sum_assignment(
            cost
        )
    )

    assigned = np.zeros_like(
        target_positions
    )

    assigned[rows] = (
        target_positions[columns]
    )

    return assigned

# DISTRIBUTED FORMATION CONTROL

def simulate_transition(
    x_start,
    x_target
):
    """
    Required distributed formation-control law:

        dx_i/dt =
        - sum_{j in N(i)}
        [(x_i-x_j) - (x_i*-x_j*)]

    Matrix form: dx/dt = -L(x-x*)

    Each agent uses relative information from its graph
    neighbors.
    """

    steps = int(
        TIME_PER_LETTER / DT
    )

    trajectory = np.zeros(
        (steps + 1, N, 2)
    )

    trajectory[0] = x_start

    x = x_start.copy()

    for k in range(steps):

        # Error in relative formation
        error = (
            x - x_target
        )

        # Distributed control
        u = -L @ error

        # Euler integration
        x = (
            x + DT * u
        )

        trajectory[k + 1] = x

    return trajectory
# BUILD NAME FORMATION SEQUENCE

current = x_initial.copy()

all_segments = []

frame_letters = []


print("\n")
print("FORMATION SEQUENCE")
print("-" * 40)


for letter_index, letter in enumerate(
    NAME.upper()
):

    print(
        f"{letter_index + 1}. Forming letter: {letter}"
    )
    # Target points for this letter
    

    target = letter_targets[
        letter
    ]
    # Assign current agents to target points

    assigned_target = assign_targets(
        current,
        target
    )
    # Move to the letter using distributed control

    segment = simulate_transition(
        current,
        assigned_target
    )
    # Add movement frames

    if len(all_segments) == 0:

        movement = segment

    else:

        movement = segment[1:]

    all_segments.append(
        movement
    )

    frame_letters.extend(
        [letter] * len(movement)
    )
    # Add a small pause after completing the letter

    pause_frames = int(
        PAUSE_TIME / DT
    )

    final_position = segment[-1]

    pause = np.repeat(
        final_position[
            np.newaxis,
            :,
            :
        ],
        pause_frames,
        axis=0
    )

    all_segments.append(
        pause
    )

    frame_letters.extend(
        [letter] * pause_frames
    )
    # Starting position for next letter

    current = (
        segment[-1].copy()
    )
# COMBINE ALL TRAJECTORIES

trajectory = np.concatenate(
    all_segments,
    axis=0
)

total_frames = len(
    trajectory
)

print("-" * 40)

print(
    "Total frames:",
    total_frames
)

print(
    "Sequence:",
    " -> ".join(
        NAME.upper()
    )
)

#CREATE FIGURE

fig, ax = plt.subplots(
    figsize=(7, 7)
)

ax.set_xlim(
    0,
    10
)

ax.set_ylim(
    0,
    10
)

ax.set_aspect(
    "equal"
)

ax.set_facecolor(
    "#0d1117"
)

fig.patch.set_facecolor(
    "#0d1117"
)

ax.axis("off")

# GRAPH EDGES

edges = list(
    G.edges()
)

edge_lines = []

for _ in edges:

    line, = ax.plot(
        [],
        [],
        color="#3d5a80",
        linewidth=0.9,
        alpha=0.6,
        zorder=1
    )

    edge_lines.append(
        line
    )
#AGENTS

agents = ax.scatter(
    [],
    [],
    s=170,
    c="#ee6c4d",
    edgecolors="white",
    linewidths=0.7,
    zorder=3
)


# TEXT

letter_text = ax.text(
    5,
    9.35,
    "",
    ha="center",
    va="center",
    fontsize=24,
    color="white",
    fontweight="bold"
)

name_text = ax.text(
    5,
    8.75,
    "",
    ha="center",
    va="center",
    fontsize=13,
    color="white"
)

#INITIALIZATION

def init():

    agents.set_offsets(
        np.zeros(
            (N, 2)
        )
    )

    for line in edge_lines:

        line.set_data(
            [],
            []
        )

    letter_text.set_text(
        ""
    )

    name_text.set_text(
        ""
    )

    return (
        edge_lines
        + [
            agents,
            letter_text,
            name_text
        ]
    )
# ANIMATION UPDATE

def update(frame):

    position = (
        trajectory[frame]
    )
    agents.set_offsets(
        position
    )

    for line, (i, j) in zip(
        edge_lines,
        edges
    ):

        line.set_data(
            [
                position[i, 0],
                position[j, 0]
            ],
            [
                position[i, 1],
                position[j, 1]
            ]
        )

    current_letter = (
        frame_letters[frame]
    )

    letter_text.set_text(
        f"Forming: {current_letter}"
    )

    name_text.set_text(
        f"{NAME.upper()}   |   N = {N}   |   p = {P}"
    )

    return (
        edge_lines
        + [
            agents,
            letter_text,
            name_text
        ]
    )
# CREATE ANIMATION

ani = animation.FuncAnimation(
    fig,
    update,
    frames=range(
        0,
        total_frames,
        2
    ),
    init_func=init,
    interval=40,
    blit=False
)

# SAVE VIDEO

OUTPUT_FILE = (
    "name_formation.mp4"
)

writer = animation.FFMpegWriter(
    fps=25,
    bitrate=2500
)

ani.save(
    OUTPUT_FILE,
    writer=writer,
    dpi=150
)

plt.close(fig)


print("\n")
print("=" * 65)
print("SIMULATION COMPLETE")
print("=" * 65)

print(
    "Video saved as:",
    OUTPUT_FILE
)
