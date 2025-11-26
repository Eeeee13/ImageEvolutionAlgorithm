import random
from dataclasses import dataclass
from math import radians
from typing import Dict, Tuple, List
from time import time

import numpy as np
from PIL import Image

# Globals for brush variants (filled by prepare_brushes)
BRUSH_VARIANTS: Dict[Tuple[float, int], np.ndarray] = {}
BRUSH_SCALES: List[float] = []
BRUSH_ANGLES: List[int] = []


@dataclass
class Individ:
    genes: list
    fitness: float = None


def prepare_brushes(brush_path: str,
                    scales: List[float] = None,
                    angles: List[int] = None) -> None:
    """
    Load brush image (RGBA) and precompute resized/rotated variants.
    Stores results in global BRUSH_VARIANTS with keys (scale, angle).
    brush_path: local path to RGBA brush PNG.
    scales: list of scale multipliers.
    angles: list of rotation angles in degrees.
    """
    global BRUSH_VARIANTS, BRUSH_SCALES, BRUSH_ANGLES
    if scales is None:
        scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    if angles is None:
        angles = list(range(0, 360, 30))

    BRUSH_VARIANTS = {}
    BRUSH_SCALES = scales
    BRUSH_ANGLES = angles

    brush_img = Image.open(brush_path).convert("RGBA")
    base_w, base_h = brush_img.size

    for s in scales:
        # resize keeping aspect ratio
        new_w = max(1, int(base_w * s))
        new_h = max(1, int(base_h * s))
        resized = brush_img.resize((new_w, new_h), Image.LANCZOS)
        for ang in angles:
            # rotate about center; expand to keep whole rotated brush
            rotated = resized.rotate(ang, resample=Image.BILINEAR, expand=True)
            arr = np.array(rotated, dtype=np.float32) / 255.0  # shape H,W,4 in 0..1
            BRUSH_VARIANTS[(s, ang)] = arr


def _find_nearest_scale_angle(scale: float, angle: float) -> Tuple[float, int]:
    """Find nearest precomputed scale and angle keys."""
    if not BRUSH_SCALES or not BRUSH_ANGLES:
        raise RuntimeError("Brush variants are not prepared. Call prepare_brushes().")
    # nearest scale
    s_nearest = min(BRUSH_SCALES, key=lambda s: abs(s - scale))
    # normalize angle to [0,360)
    ang_norm = int(round(angle)) % 360
    a_nearest = min(BRUSH_ANGLES,
                    key=lambda a: min(abs(a - ang_norm), 360 - abs(a - ang_norm)))
    return s_nearest, a_nearest


def load_target_image(path: str, img_size: int):
    """Load target image as uint8 numpy array (H,W,3)."""
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def random_gene(img_size: int):
    """Generate a random brush-stroke gene."""
    x = random.randint(0, img_size - 1)
    y = random.randint(0, img_size - 1)
    scale = random.uniform(0.5, 1.5)
    rotation = random.uniform(0.0, 359.9)
    r_mul = random.uniform(0.8, 1.2)
    g_mul = random.uniform(0.8, 1.2)
    b_mul = random.uniform(0.8, 1.2)
    alpha = random.uniform(0.1, 0.9)
    return (x, y, scale, rotation, r_mul, g_mul, b_mul, alpha)


def init_population(pop_size: int, n_genes: int, img_size: int):
    population = []
    for _ in range(pop_size):
        genes = [random_gene(img_size) for _ in range(n_genes)]
        population.append(Individ(genes=genes))
    return population


def mutate(individ: Individ, mutation_rate: float, img_size: int) -> Individ:
    """Mutate a brush-stroke individ; returns new Individ."""
    new_genes = []
    for gene in individ.genes:
        x, y, scale, rotation, r_mul, g_mul, b_mul, alpha = gene
        if random.random() < mutation_rate:
            x += random.randint(-20, 20)
            y += random.randint(-20, 20)
            scale += random.uniform(-0.2, 0.2)
            rotation += random.uniform(-30.0, 30.0)
            r_mul += random.uniform(-0.1, 0.1)
            g_mul += random.uniform(-0.1, 0.1)
            b_mul += random.uniform(-0.1, 0.1)
            alpha += random.uniform(-0.05, 0.05)

            x = max(0, min(img_size - 1, int(x)))
            y = max(0, min(img_size - 1, int(y)))
            scale = max(0.1, min(3.0, scale))
            rotation = rotation % 360.0
            r_mul = max(0.2, min(2.0, r_mul))
            g_mul = max(0.2, min(2.0, g_mul))
            b_mul = max(0.2, min(2.0, b_mul))
            alpha = max(0.0, min(1.0, alpha))

        new_genes.append((x, y, scale, rotation, r_mul, g_mul, b_mul, alpha))
    return Individ(genes=new_genes)

def downscale(img, factor=8):
    h, w, c = img.shape
    new_h, new_w = h // factor, w // factor
    img = img[:new_h * factor, :new_w * factor]
    return img.reshape(new_h, factor, new_w, factor, c).mean(axis=(1, 3))


def image_histogram(img, bins=32):
    hist = []
    for ch in range(3):
        h, _ = np.histogram(img[:, :, ch], bins=bins, range=(0, 255))
        hist.append(h.astype(np.float32))
    return np.concatenate(hist)


def select_parent(population, k=3):
    sample = random.sample(population, k)
    return max(sample, key=lambda ind: ind.fitness)

def crossover(p1: Individ, p2: Individ, alpha: float = 1.6) -> Individ:
    """
    Biased crossover:
    - Вероятность взять ген у сильного родителя выше.
    - Усиление через (f1 / (f1 + f2))**alpha.
    - clamp в [0.55, 0.90] для предотвращения сжатия разнообразия.
    """
    f1 = p1.fitness
    f2 = p2.fitness

    # Avoid division issues
    if f1 < 1e-12:
        f1 = 1e-12
    if f2 < 1e-12:
        f2 = 1e-12

    # Probability p = (f1 / (f1 + f2))^alpha
    raw_p = (f1 / (f1 + f2)) ** alpha

    # Clamp to ensure exploration continues
    p = max(0.55, min(0.90, raw_p))

    # If p1 worse, invert probability
    if f2 > f1:
        p = 1.0 - p

    child_genes = []
    for g1, g2 in zip(p1.genes, p2.genes):
        if random.random() < p:
            child_genes.append(g1)
        else:
            child_genes.append(g2)

    return Individ(genes=child_genes)


import random

import numpy as np
import time
from statistics import mean


def _compute_gradient_map(img):
    """Compute gradient magnitude map of an RGB uint8 image (float32)."""
    gray = img.astype(np.float32).mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    grad = np.hypot(gx, gy)
    return grad


def compute_fitness(individ_img, target_img, gradient_weight=0.0):
    """
    Compute fitness combining downscaled MSE and color histograms.
    Optionally include gradient-structure component (gradient_weight in [0,1]).
    """
    # downscale for MSE/component
    small_ind = downscale(individ_img)
    small_tgt = downscale(target_img)

    mse = np.mean((small_ind - small_tgt) ** 2)
    score_mse = 1.0 / (1.0 + mse)

    hist_ind = image_histogram(individ_img)
    hist_tgt = image_histogram(target_img)
    # normalize histograms to densities to reduce scale effects
    hist_ind = hist_ind / (np.sum(hist_ind) + 1e-12)
    hist_tgt = hist_tgt / (np.sum(hist_tgt) + 1e-12)

    diff = np.sum(np.abs(hist_ind - hist_tgt))
    score_hist = 1.0 / (1.0 + diff)

    base_score = 0.7 * score_mse + 0.3 * score_hist

    if gradient_weight and gradient_weight > 0.0:
        g_ind = _compute_gradient_map(individ_img)
        g_tgt = _compute_gradient_map(target_img)
        g_small_ind = downscale(g_ind[..., None], factor=8)[..., 0]
        g_small_tgt = downscale(g_tgt[..., None], factor=8)[..., 0]
        g_diff = np.mean(np.abs(g_small_ind - g_small_tgt))
        score_grad = 1.0 / (1.0 + g_diff)
        # Mix base score and gradient score
        fitness = (1.0 - gradient_weight) * base_score + gradient_weight * score_grad
        return float(fitness)

    return float(base_score)


def adjust_params(best_history,
                  std_f,
                  ma_window,
                  mutation_rate_min,
                  mutation_rate_max,
                  pop_min,
                  pop_max,
                  mutation_rate_current,
                  pop_current):
    """
    Compute new mutation_rate and pop_size using sliding mean of best_history.
    Returns (new_mutation_rate, new_pop_size).
    """
    if len(best_history) < ma_window:
        return mutation_rate_current, pop_current

    F_ma = mean(best_history[-ma_window:])  # in 0..1

    # linear interpolation for mutation_rate: high fitness -> low mutation
    new_mutation = mutation_rate_max * (1.0 - F_ma) + mutation_rate_min * F_ma

    # pop size: high fitness -> smaller population
    new_pop = int(pop_min + (pop_max - pop_min) * (1.0 - F_ma))
    new_pop = max(pop_min, min(pop_max, new_pop))

    # diversity adjustment: if std too small, encourage exploration
    std_threshold = 0.01
    if std_f < std_threshold:
        new_mutation = min(mutation_rate_max, new_mutation * 1.5)
        new_pop = min(pop_max, int(new_pop * 1.5))

    return float(new_mutation), int(new_pop)


def run_ga(target_img,
           pop_size_base,
           n_genes,
           img_size,
           generations,
           mutation_rate_base,
           run_id=None,
           # adaptive params
           mutation_rate_min=0.02,
           mutation_rate_max=0.4,
           pop_size_min=16,
           pop_size_max=80,
           adapt_every=5,
           ma_window=7,
           elitism=2,
           gradient_weight=0.0,
           save_every=10):
    """
    Adaptive GA run. Returns final population and logs.
    - gradient_weight: if >0, compute_fitness includes gradient term.
    - save_every: frequency (generations) to save intermediate best image if run_id provided.
    """

    # init adaptive variables
    pop_size = int(pop_size_base)
    mutation_rate = float(mutation_rate_base)

    # initialize population
    population = init_population(pop_size, n_genes, img_size)

    # compute initial fitness
    for ind in population:
        img = render_individual(ind, img_size)
        ind.fitness = compute_fitness(img, target_img, gradient_weight=gradient_weight)

    best_per_gen = []
    avg_per_gen = []
    std_per_gen = []
    mutation_history = []
    pop_history = []
    times_per_gen = []

    start_total = time.perf_counter()

    for gen in range(1, generations + 1):
        t0 = time.perf_counter()
        new_population = []

        # elitism: copy top-k
        sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        elites = sorted_pop[:elitism]
        for e in elites:
            # shallow copy genes; fitness will be preserved
            new_population.append(Individ(genes=list(e.genes), fitness=e.fitness))

        # generate rest of new population
        while len(new_population) < pop_size:
            p1 = select_parent(population)
            p2 = select_parent(population)
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate, img_size)
            img = render_individual(child, img_size)
            child.fitness = compute_fitness(img, target_img, gradient_weight=gradient_weight)
            new_population.append(child)

        population = new_population

        fitness_values = [ind.fitness for ind in population]
        best_fit = max(fitness_values)
        avg_fit = float(sum(fitness_values) / len(fitness_values))
        std_fit = float(np.std(fitness_values))

        best_per_gen.append(best_fit)
        avg_per_gen.append(avg_fit)
        std_per_gen.append(std_fit)
        mutation_history.append(mutation_rate)
        pop_history.append(pop_size)

        # save intermediate best image periodically
        if run_id is not None and (gen % save_every == 0 or gen == 1 or gen == generations):
            best_ind = max(population, key=lambda ind: ind.fitness)
            best_img = render_individual(best_ind, img_size)
            Image.fromarray(best_img).save(f"run_{run_id}_gen_{gen}.jpg", quality=95)

        # adapt parameters every adapt_every gens
        if gen % adapt_every == 0 and gen >= ma_window:
            mutation_rate, new_pop = adjust_params(
                best_history=best_per_gen,
                std_f=std_fit,
                ma_window=ma_window,
                mutation_rate_min=mutation_rate_min,
                mutation_rate_max=mutation_rate_max,
                pop_min=pop_size_min,
                pop_max=pop_size_max,
                mutation_rate_current=mutation_rate,
                pop_current=pop_size
            )
            pop_size = int(new_pop)

        t1 = time.perf_counter()
        times_per_gen.append(t1 - t0)

        # if nearing time budget, can break early (optional)
        # elapsed = time.perf_counter() - start_total
        # if elapsed > 60 * 9.5:
        #     break

    total_time = time.perf_counter() - start_total

    logs = {
        "best_per_gen": best_per_gen,
        "avg_per_gen": avg_per_gen,
        "std_per_gen": std_per_gen,
        "mutation_history": mutation_history,
        "pop_history": pop_history,
        "times_per_gen": times_per_gen,
        "total_time": total_time
    }

    return population, logs




def save_best_individuals(population, img_size, top_k, base_filename):
    sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
    best = sorted_pop[:top_k]

    for i, ind in enumerate(best):
        img = render_individual(ind, img_size)
        Image.fromarray(img).save(f"{base_filename}_{i+1}.jpg", quality=95)


def render_individual(individ: Individ, img_size: int) -> np.ndarray:
    """
    Render individ composed of brush strokes.
    Returns uint8 image (img_size, img_size, 3).
    Requires prepare_brushes(...) was called beforehand.
    """
    canvas = np.zeros((img_size, img_size, 3), dtype=np.float32)

    for gene in individ.genes:
        x, y, scale, rotation, r_mul, g_mul, b_mul, alpha = gene
        # select nearest precomputed variant
        s_key, a_key = _find_nearest_scale_angle(scale, rotation)
        brush = BRUSH_VARIANTS.get((s_key, a_key))
        if brush is None:
            continue

        # brush: HxWx4 in 0..1
        bh, bw, _ = brush.shape
        # top-left on canvas
        left = int(x - bw // 2)
        top = int(y - bh // 2)

        # compute overlapping region
        c_x0 = max(0, left)
        c_y0 = max(0, top)
        c_x1 = min(img_size, left + bw)
        c_y1 = min(img_size, top + bh)

        if c_x0 >= c_x1 or c_y0 >= c_y1:
            continue

        b_x0 = c_x0 - left
        b_y0 = c_y0 - top
        b_x1 = b_x0 + (c_x1 - c_x0)
        b_y1 = b_y0 + (c_y1 - c_y0)

        brush_region = brush[b_y0:b_y1, b_x0:b_x1]  # shape rh,rw,4
        if brush_region.size == 0:
            continue

        brush_rgb = brush_region[..., :3]  # 0..1
        brush_a = brush_region[..., 3]  # 0..1

        # apply color multipliers and global alpha
        color_mul = np.array([r_mul, g_mul, b_mul], dtype=np.float32)
        src_rgb = brush_rgb * color_mul[None, None, :]
        src_alpha = brush_a * alpha  # 0..1

        dst = canvas[c_y0:c_y1, c_x0:c_x1]  # float32

        # composite: out = src * alpha + dst * (1 - alpha)
        src_alpha_3 = src_alpha[..., None]
        out = src_rgb * src_alpha_3 + dst * (1.0 - src_alpha_3)
        canvas[c_y0:c_y1, c_x0:c_x1] = out

    return np.clip(canvas * 255.0, 0, 255).astype(np.uint8)


def main():
    IMG_SIZE = 512
    POP_SIZE = 30
    GENERATIONS = 100
    N_GENES = 500
    MUTATION_RATE = 0.4

    BASE_NAME = "NameSurname"
    INPUT_INDEX = 1
    INPUT_PATH = f"input{INPUT_INDEX}.jpg"

    # 1. Подготовка кистей
    prepare_brushes(
        "whiteBrush.png",
        scales=[0.01, 0.05, 0.1, 0.25],
        angles=list(range(0, 360, 30))
    )

    # 2. Загрузка целевого изображения
    target_img = load_target_image(INPUT_PATH, IMG_SIZE)

    # 3. Три независимых прогона GA
    for run_id in range(1, 4):
        population, logs = run_ga(
        target_img=target_img,
        pop_size_base=POP_SIZE,
        n_genes=N_GENES,
        img_size=IMG_SIZE,
        generations=GENERATIONS,
        mutation_rate_base=MUTATION_RATE,
        run_id=run_id,
        mutation_rate_min=0.02,
        mutation_rate_max=0.7,
        pop_size_min=12,
        pop_size_max=80,
        adapt_every=5,
        ma_window=7,
        elitism=3,
        gradient_weight=0.1,  # optional, set 0.0 to disable
        save_every=10
) 
    

        base_filename = f"{BASE_NAME}Output{INPUT_INDEX}_{run_id}"
        save_best_individuals(population, IMG_SIZE, 5, base_filename)


if __name__ == "__main__":
    main()
