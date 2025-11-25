import numpy as np
from PIL import Image, ImageDraw
import random
import time
from typing import List, Tuple
import os

import random
from dataclasses import dataclass


@dataclass
class Individ:
    genes: list
    fitness: float = None


def random_gene(img_size):
    shape_type = 0  # 0 = ellipse (пока один тип, добавим позже)
    x = random.randint(0, img_size - 1)
    y = random.randint(0, img_size - 1)
    w = random.randint(5, img_size // 4)
    h = random.randint(5, img_size // 4)
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    a = random.uniform(0.1, 1.0)
    return (shape_type, x, y, w, h, r, g, b, a)


def init_population(pop_size, n_genes, img_size):
    population = []
    for _ in range(pop_size):
        genes = [random_gene(img_size) for _ in range(n_genes)]
        population.append(Individ(genes=genes))
    return population

def render_individual(individ, img_size):
    img = np.zeros((img_size, img_size, 3), dtype=np.float32)
    yy, xx = np.ogrid[:img_size, :img_size]

    for gene in individ.genes:
        (
            shape_type, x, y, w, h,
            r, g, b, a
        ) = gene

        if w <= 0 or h <= 0:
            continue

        dx = (xx - x) / w
        dy = (yy - y) / h
        mask = dx * dx + dy * dy <= 1.0

        if not np.any(mask):
            continue

        color = np.array([r, g, b], dtype=np.float32)
        img[mask] = img[mask] * (1.0 - a) + color * a

    return np.clip(img, 0, 255).astype(np.uint8)

def compute_fitness(individ_img, target_img):
    small_ind = downscale(individ_img)
    small_tgt = downscale(target_img)

    mse = np.mean((small_ind - small_tgt) ** 2)
    score_mse = 1.0 / (1.0 + mse)

    hist_ind = image_histogram(individ_img)
    hist_tgt = image_histogram(target_img)

    diff = np.sum(np.abs(hist_ind - hist_tgt))
    score_hist = 1.0 / (1.0 + diff)

    return 0.7 * score_mse + 0.3 * score_hist

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



def crossover(parent1, parent2):
    child_genes = []
    for g1, g2 in zip(parent1.genes, parent2.genes):
        if random.random() < 0.5:
            child_genes.append(g1)
        else:
            child_genes.append(g2)
    return Individ(genes=child_genes)

import random


def mutate(individ, mutation_rate, img_size):
    new_genes = []

    for gene in individ.genes:
        (
            shape_type, x, y, w, h,
            r, g, b, a
        ) = gene

        if random.random() < mutation_rate:
            x += random.randint(-15, 15)
            y += random.randint(-15, 15)
            w += random.randint(-10, 10)
            h += random.randint(-10, 10)

            r += random.randint(-20, 20)
            g += random.randint(-20, 20)
            b += random.randint(-20, 20)

            a += random.uniform(-0.1, 0.1)

            x = max(0, min(img_size - 1, x))
            y = max(0, min(img_size - 1, y))
            w = max(1, min(img_size, w))
            h = max(1, min(img_size, h))

            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            a = max(0.0, min(1.0, a))

        new_genes.append(
            (shape_type, x, y, w, h, r, g, b, a)
        )

    return Individ(genes=new_genes)

def select_parent(population, k=3):
    sample = random.sample(population, k)
    return max(sample, key=lambda ind: ind.fitness)


def run_ga(target_img,
           pop_size,
           n_genes,
           img_size,
           generations,
           mutation_rate):

    population = init_population(pop_size, n_genes, img_size)

    # вычисляем фитнесс начальной популяции
    for ind in population:
        img = render_individual(ind, img_size)
        ind.fitness = compute_fitness(img, target_img)

    best_per_gen = []
    avg_per_gen = []

    for i in range(generations):
        new_population = []

        for j in range(pop_size):
            p1 = select_parent(population)
            p2 = select_parent(population)

            child = crossover(p1, p2)
            child = mutate(child, mutation_rate/(i+1), img_size)

            img = render_individual(child, img_size)
            child.fitness = compute_fitness(img, target_img)

            new_population.append(child)

        population = new_population
        base = f'{i}_{j}'
        save_best_individuals(population, 512, 5, base)

        fitness_values = [ind.fitness for ind in population]
        best_per_gen.append(max(fitness_values))
        avg_per_gen.append(sum(fitness_values) / len(fitness_values))

    return population, best_per_gen, avg_per_gen

def load_target_image(path, img_size):
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def save_best_individuals(population, img_size, top_k, base_filename):
    sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
    best = sorted_pop[:top_k]

    for i, ind in enumerate(best):
        img = render_individual(ind, img_size)
        Image.fromarray(img).save(f"{base_filename}_{i+1}.jpg", quality=95)


def main():
    IMG_SIZE = 512
    POP_SIZE = 30
    GENERATIONS = 200
    N_GENES = 100
    MUTATION_RATE = 1

    BASE_NAME = "NameSurname"
    INPUT_INDEX = 1
    INPUT_PATH = f"input{INPUT_INDEX}.jpg"

    target_img = load_target_image(INPUT_PATH, IMG_SIZE)

    for run_id in range(1, 4):
        population, best_log, avg_log = run_ga(
            target_img=target_img,
            pop_size=POP_SIZE,
            n_genes=N_GENES,
            img_size=IMG_SIZE,
            generations=GENERATIONS,
            mutation_rate=MUTATION_RATE
        )

        base_filename = f"{BASE_NAME}Output{INPUT_INDEX}_{run_id}"
        save_best_individuals(population, IMG_SIZE, 5, base_filename)


main()

