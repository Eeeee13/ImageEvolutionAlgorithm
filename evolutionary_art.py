import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
import time
import random
from typing import List, Tuple
import os
from skimage.metrics import structural_similarity as ssim

class BrushStrokeEA:
    def __init__(self, target_path: str, brush_path: str, output_prefix: str):
        """
        Evolutionary Algorithm for image generation using brush strokes with layered approach
        
        Args:
            target_path: Path to target image
            brush_path: Path to brush stroke PNG
            output_prefix: Prefix for output files (NameSurname)
        """
        # Load target image
        self.target = cv2.imread(target_path)
        self.target = cv2.resize(self.target, (512, 512))
        self.target_gray = cv2.cvtColor(self.target, cv2.COLOR_BGR2GRAY)
        self.target_edges = cv2.Canny(self.target_gray, 50, 150)
        
        # Compute gradient direction for brush orientation
        sobelx = cv2.Sobel(self.target_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(self.target_gray, cv2.CV_64F, 0, 1, ksize=3)
        self.gradient_direction = np.arctan2(sobely, sobelx) * 180 / np.pi
        
        # Precompute target histograms for each channel
        self.target_hist = [cv2.calcHist([self.target], [i], None, [256], [0, 256]) for i in range(3)]
        
        # Load and preprocess brush stroke
        self.brush_original = Image.open(brush_path).convert('RGBA')
        self.brush_cache = {}
        self._preload_brushes()
        
        self.output_prefix = output_prefix
        self.image_index = os.path.basename(target_path).replace('input', '').replace('.jpg', '')
        
        # EA parameters
        self.population_size = 30
        self.elitism_count = 5
        self.crossover_rate = 0.8
        self.mutation_rate = 0.3
        
        # Fitness weights (added SSIM)
        self.w_mse = 0.25
        self.w_hist = 0.25
        self.w_edge = 0.25
        self.w_ssim = 0.25
        
        # Layered approach with reduced small strokes and adjusted alpha
        self.layers = [
            (10, (150, 300), 60, "background", (180, 255)),      # Layer 1: Background (opaque)
            (30, (60, 90), 80, "base", (150, 230)),             # Layer 2: Base shapes
            (100, (30, 50), 100, "medium", (130, 200)),         # Layer 3: Medium details
            (200, (15, 25), 120, "details", (100, 170)),        # Layer 4: Fine details (reduced from 250)
            (150, (8, 15), 100, "refinement", (80, 140))        # Layer 5: Refinement (reduced from 400)
        ]
        
        self.accumulated_strokes = []
        
        # Early stopping parameters
        self.patience = 15  # Stop if no improvement for 15 generations
        
    def _preload_brushes(self):
        """Preload brush strokes of different sizes for speed"""
        sizes = [8, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 150]
        for size in sizes:
            aspect_ratio = self.brush_original.width / self.brush_original.height
            new_width = int(size * aspect_ratio)
            brush_resized = self.brush_original.resize((new_width, size), Image.LANCZOS)
            self.brush_cache[size] = brush_resized
    
    def _get_brush(self, size: int) -> Image.Image:
        """Get cached or create brush of specified size"""
        if size in self.brush_cache:
            return self.brush_cache[size]
        
        closest = min(self.brush_cache.keys(), key=lambda x: abs(x - size))
        return self.brush_cache[closest]
    
    def sample_color_from_region(self, x: int, y: int, radius: int = 10) -> Tuple[int, int, int]:
        """Sample average color from region around point"""
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(512, x + radius)
        y2 = min(512, y + radius)
        
        region = self.target[y1:y2, x1:x2]
        avg_color = np.mean(region, axis=(0, 1))
        return tuple(map(int, avg_color[::-1]))  # BGR to RGB
    
    def get_gradient_angle(self, x: int, y: int) -> float:
        """Get gradient direction at point for brush orientation"""
        x = np.clip(x, 0, 511)
        y = np.clip(y, 0, 511)
        return self.gradient_direction[int(y), int(x)]
    
    def create_smart_individual(self, num_strokes: int, size_range: Tuple[int, int], 
                               layer_type: str, alpha_range: Tuple[int, int]) -> np.ndarray:
        """
        Create individual with smart initialization based on target image
        Each stroke: [x, y, r, g, b, size, rotation, alpha]
        """
        individual = np.zeros((num_strokes, 8))
        
        if layer_type == "background":
            # Background: cover entire canvas with large strokes
            grid_size = int(np.sqrt(num_strokes)) + 1
            spacing = 512 // grid_size
            idx = 0
            
            for i in range(grid_size):
                for j in range(grid_size):
                    if idx >= num_strokes:
                        break
                    x = min(i * spacing + spacing // 2, 511)
                    y = min(j * spacing + spacing // 2, 511)
                    
                    individual[idx, 0] = x + np.random.randint(-spacing//3, spacing//3)
                    individual[idx, 1] = y + np.random.randint(-spacing//3, spacing//3)
                    
                    # Sample color from target
                    r, g, b = self.sample_color_from_region(int(individual[idx, 0]), 
                                                           int(individual[idx, 1]), radius=30)
                    individual[idx, 2:5] = [r, g, b]
                    individual[idx, 5] = np.random.randint(size_range[0], size_range[1])
                    
                    # Use gradient direction for rotation
                    angle = self.get_gradient_angle(individual[idx, 0], individual[idx, 1])
                    individual[idx, 6] = (angle + np.random.randint(-20, 20)) % 360
                    
                    individual[idx, 7] = np.random.randint(alpha_range[0], alpha_range[1])
                    idx += 1
                    
        elif layer_type in ["base", "medium"]:
            # Base/Medium: focus on edges and color regions
            for i in range(num_strokes):
                # Mix of edge-based and random placement
                if random.random() < 0.6:  # 60% on edges
                    edge_points = np.argwhere(self.target_edges > 100)
                    if len(edge_points) > 0:
                        point = edge_points[np.random.randint(len(edge_points))]
                        individual[i, 1], individual[i, 0] = point  # y, x
                    else:
                        individual[i, 0] = np.random.randint(0, 512)
                        individual[i, 1] = np.random.randint(0, 512)
                else:  # 40% random
                    individual[i, 0] = np.random.randint(0, 512)
                    individual[i, 1] = np.random.randint(0, 512)
                
                # Sample color from target
                r, g, b = self.sample_color_from_region(int(individual[i, 0]), 
                                                       int(individual[i, 1]), radius=15)
                individual[i, 2:5] = [r, g, b]
                individual[i, 5] = np.random.randint(size_range[0], size_range[1])
                
                # Use gradient direction for rotation
                angle = self.get_gradient_angle(individual[i, 0], individual[i, 1])
                individual[i, 6] = (angle + np.random.randint(-30, 30)) % 360
                
                individual[i, 7] = np.random.randint(alpha_range[0], alpha_range[1])
                
        else:  # details, refinement
            # Detail layers: more random but still color-aware
            individual[:, 0] = np.random.randint(0, 512, num_strokes)
            individual[:, 1] = np.random.randint(0, 512, num_strokes)
            
            for i in range(num_strokes):
                r, g, b = self.sample_color_from_region(int(individual[i, 0]), 
                                                       int(individual[i, 1]), radius=8)
                individual[i, 2:5] = [r, g, b]
                
                # Use gradient direction for rotation
                angle = self.get_gradient_angle(individual[i, 0], individual[i, 1])
                individual[i, 6] = (angle + np.random.randint(-45, 45)) % 360
            
            individual[:, 5] = np.random.randint(size_range[0], size_range[1], num_strokes)
            individual[:, 7] = np.random.randint(alpha_range[0], alpha_range[1], num_strokes)
        
        return individual
    
    def render_strokes(self, strokes_list: List[np.ndarray]) -> np.ndarray:
        """Render multiple layers of strokes"""
        canvas = Image.new('RGB', (512, 512), (255, 255, 255))
        
        # Render all layers in order
        for strokes in strokes_list:
            for stroke in strokes:
                x, y, r, g, b, size, rotation, alpha = stroke
                
                brush = self._get_brush(int(size))
                brush = brush.rotate(rotation, expand=True, resample=Image.BILINEAR)
                
                colored = Image.new('RGBA', brush.size, (int(r), int(g), int(b), int(alpha)))
                colored.putalpha(brush.split()[3])
                
                paste_x = int(x - brush.width // 2)
                paste_y = int(y - brush.height // 2)
                
                try:
                    canvas.paste(colored, (paste_x, paste_y), colored)
                except:
                    pass
        
        return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    
    def fitness(self, current_layer: np.ndarray) -> float:
        """
        Calculate fitness using MSE, histogram, edge similarity, and SSIM
        Lower is better
        """
        # Render all accumulated layers + current layer
        all_strokes = self.accumulated_strokes + [current_layer]
        rendered = self.render_strokes(all_strokes)
        
        # MSE
        mse = np.mean((self.target.astype(float) - rendered.astype(float)) ** 2)
        
        # Histogram comparison
        hist_diff = 0
        for i in range(3):
            hist = cv2.calcHist([rendered], [i], None, [256], [0, 256])
            hist_diff += cv2.compareHist(self.target_hist[i], hist, cv2.HISTCMP_CHISQR)
        hist_diff /= 3
        
        # Edge similarity
        rendered_gray = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY)
        rendered_edges = cv2.Canny(rendered_gray, 50, 150)
        edge_diff = np.mean((self.target_edges.astype(float) - rendered_edges.astype(float)) ** 2)
        
        # SSIM (Structural Similarity) - convert to dissimilarity (1 - ssim)
        ssim_score = ssim(self.target_gray, rendered_gray, data_range=255)
        ssim_dissimilarity = (1 - ssim_score) * 10000  # Scale to similar range as MSE
        
        # Weighted fitness
        fitness_value = (self.w_mse * mse + 
                        self.w_hist * hist_diff + 
                        self.w_edge * edge_diff + 
                        self.w_ssim * ssim_dissimilarity)
        
        return fitness_value
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Single-point crossover"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        point = random.randint(1, len(parent1) - 1)
        child1 = np.vstack([parent1[:point], parent2[point:]])
        child2 = np.vstack([parent2[:point], parent1[point:]])
        
        return child1, child2
    
    def mutate(self, individual: np.ndarray, size_range: Tuple[int, int], 
               alpha_range: Tuple[int, int], mutation_rate: float) -> np.ndarray:
        """Mutate individual parameters with color sampling and gradient-based rotation"""
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                gene = random.randint(0, 7)
                
                if gene == 0:  # x
                    mutated[i, 0] = np.clip(mutated[i, 0] + np.random.randint(-30, 30), 0, 511)
                elif gene == 1:  # y
                    mutated[i, 1] = np.clip(mutated[i, 1] + np.random.randint(-30, 30), 0, 511)
                elif gene in [2, 3, 4]:  # r, g, b - resample from target
                    if random.random() < 0.5:
                        r, g, b = self.sample_color_from_region(int(mutated[i, 0]), 
                                                               int(mutated[i, 1]), radius=10)
                        mutated[i, 2:5] = [r, g, b]
                    else:
                        mutated[i, gene] = np.clip(mutated[i, gene] + np.random.randint(-20, 20), 0, 255)
                elif gene == 5:  # size
                    mutated[i, 5] = np.clip(mutated[i, 5] + np.random.randint(-5, 5), 
                                           size_range[0], size_range[1])
                elif gene == 6:  # rotation - use gradient with perturbation
                    if random.random() < 0.3:
                        angle = self.get_gradient_angle(mutated[i, 0], mutated[i, 1])
                        mutated[i, 6] = (angle + np.random.randint(-30, 30)) % 360
                    else:
                        mutated[i, 6] = (mutated[i, 6] + np.random.randint(-45, 45)) % 360
                elif gene == 7:  # alpha
                    mutated[i, 7] = np.clip(mutated[i, 7] + np.random.randint(-20, 20), 
                                           alpha_range[0], alpha_range[1])
        
        return mutated
    
    def evolve_layer(self, layer_num: int, num_strokes: int, size_range: Tuple[int, int], 
                     generations: int, layer_type: str, alpha_range: Tuple[int, int]) -> Tuple:
        """Evolve one layer with early stopping"""
        print(f"\n=== Layer {layer_num} ({layer_type}): {num_strokes} strokes, "
              f"size {size_range}, max {generations} generations ===")
        
        # Initialize population with smart placement
        population = [self.create_smart_individual(num_strokes, size_range, layer_type, alpha_range) 
                     for _ in range(self.population_size)]
        
        avg_fitness_history = []
        max_fitness_history = []
        
        best_fitness = float('inf')
        patience_counter = 0
        
        for gen in range(generations):
            # Evaluate fitness
            fitness_scores = [self.fitness(ind) for ind in population]
            
            avg_fitness = np.mean(fitness_scores)
            min_fitness = np.min(fitness_scores)
            avg_fitness_history.append(avg_fitness)
            max_fitness_history.append(min_fitness)
            
            # Early stopping check
            if min_fitness < best_fitness - 1.0:  # Improvement threshold
                best_fitness = min_fitness
                patience_counter = 0
            else:
                patience_counter += 1
            
            if gen % 10 == 0:
                print(f"Gen {gen}/{generations}: Avg={avg_fitness:.2f}, Best={min_fitness:.2f}, "
                      f"Patience={patience_counter}/{self.patience}")
            
            # Early stopping
            if patience_counter >= self.patience:
                print(f"Early stopping at generation {gen} (no improvement for {self.patience} gens)")
                break
            
            # Selection
            sorted_indices = np.argsort(fitness_scores)
            sorted_population = [population[i] for i in sorted_indices]
            
            # Elitism
            new_population = sorted_population[:self.elitism_count]
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = sorted_population[random.randint(0, self.population_size // 2)]
                parent2 = sorted_population[random.randint(0, self.population_size // 2)]
                
                child1, child2 = self.crossover(parent1, parent2)
                
                # Adaptive mutation
                adaptive_rate = self.mutation_rate * (1 - 0.5 * gen / generations)
                child1 = self.mutate(child1, size_range, alpha_range, adaptive_rate)
                child2 = self.mutate(child2, size_range, alpha_range, adaptive_rate)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        # Return best individual
        fitness_scores = [self.fitness(ind) for ind in population]
        best_idx = np.argmin(fitness_scores)
        
        print(f"Layer {layer_num} complete. Best fitness: {fitness_scores[best_idx]:.2f}")
        
        return population[best_idx], avg_fitness_history, max_fitness_history
    
    def run(self, run_number: int):
        """Run complete evolutionary algorithm with layered approach"""
        print(f"\n{'='*70}")
        print(f"Starting EA Run {run_number} - Layered Approach with Improvements")
        print(f"{'='*70}")
        
        start_time = time.time()
        self.accumulated_strokes = []
        
        all_avg_histories = []
        all_max_histories = []
        
        # Evolve each layer sequentially
        for layer_num, (num_strokes, size_range, generations, layer_type, alpha_range) in enumerate(self.layers, 1):
            best_layer, avg_hist, max_hist = self.evolve_layer(
                layer_num, num_strokes, size_range, generations, layer_type, alpha_range
            )
            
            # Add this layer to accumulated strokes
            self.accumulated_strokes.append(best_layer)
            all_avg_histories.append(avg_hist)
            all_max_histories.append(max_hist)
            
            # Show intermediate result
            intermediate = self.render_strokes(self.accumulated_strokes)
            cv2.imwrite(f"temp_layer{layer_num}_run{run_number}.jpg", intermediate)
        
        elapsed_time = time.time() - start_time
        print(f"\nTotal time: {elapsed_time/60:.2f} minutes")
        
        # Generate final variations by slightly mutating the best result
        final_results = []
        
        # Best result (no mutation)
        final_results.append(self.render_strokes(self.accumulated_strokes))
        
        # 4 variations with slight mutations
        for i in range(4):
            mutated_strokes = []
            for layer_idx, layer in enumerate(self.accumulated_strokes):
                size_range = (int(np.min(layer[:, 5])), int(np.max(layer[:, 5])))
                alpha_range = (int(np.min(layer[:, 7])), int(np.max(layer[:, 7])))
                mutated = self.mutate(layer.copy(), size_range, alpha_range, 0.05)
                mutated_strokes.append(mutated)
            final_results.append(self.render_strokes(mutated_strokes))
        
        # Save top 5 results
        for i, result in enumerate(final_results, 1):
            output_path = f"{self.output_prefix}Output{self.image_index}_{run_number}_{i}.jpg"
            cv2.imwrite(output_path, result)
            print(f"Saved: {output_path}")
        
        return elapsed_time, all_avg_histories, all_max_histories


# Main execution
if __name__ == "__main__":
    INPUT_IMAGE = "input5.jpg"
    BRUSH_STROKE = "brush.png"
    OUTPUT_PREFIX = "E"
    NUM_RUNS = 1
    
    for run in range(1, NUM_RUNS + 1):
        ea = BrushStrokeEA(INPUT_IMAGE, BRUSH_STROKE, OUTPUT_PREFIX)
        elapsed, avg_histories, max_histories = ea.run(run)
        
        print(f"\n{'='*70}")
        print(f"Run {run} completed in {elapsed/60:.2f} minutes")
        print(f"{'='*70}\n")