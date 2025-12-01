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
        Multi-Resolution Evolutionary Algorithm with Greedy Refinement
        
        Args:
            target_path: Path to target image
            brush_path: Path to brush stroke PNG
            output_prefix: Prefix for output files (NameSurname)
        """
        # Load original target image
        self.target_original = cv2.imread(target_path)
        self.target_original = cv2.resize(self.target_original, (512, 512))
        
        # Current resolution targets (will be set per phase)
        self.target = None
        self.target_gray = None
        self.target_edges = None
        self.target_hist = None
        self.gradient_direction = None
        self.current_resolution = None

        self.run_folder = None
        
        # Load and preprocess brush stroke
        self.brush_original = Image.open(brush_path).convert('RGBA')
        self.brush_cache = {}
        self._preload_brushes()
        self.canvas_cache = None  # PIL Image
        self.canvas_cache_cv = None  # CV2 array для fitness
        
        self.output_prefix = output_prefix
        self.image_index = os.path.basename(target_path).replace('input', '').replace('.jpg', '')
        
        # EA parameters
        self.population_size = 25
        self.elitism_count = 5
        self.crossover_rate = 0.8
        self.mutation_rate = 0.3
        
        # Fitness weights
        self.w_mse = 0.25
        self.w_hist = 0.4
        self.w_edge = 0.15
        self.w_ssim = 0.2
        
        # Multi-resolution phases: (resolution, layers_config, greedy_iterations)
        # Each layer: (num_strokes, size_range, generations, layer_type, alpha_range)
        self.phases = [
            # Phase 0: 64x64 - Form and Color
            (64, [
                (50, (20, 30), 40, "background", (200, 255)),
            ], 10),
            # Phase 1: 128x128 - Form and Color
            (128, [
                (12, (20, 30), 40, "background", (200, 255)),
                (15, (12, 20), 50, "base", (160, 220)),
                (60, (6, 12), 60, "structure", (130, 190))
            ], 30),
            
            # Phase 2: 256x256 - Structure
            (256, [
                (100, (10, 18), 70, "medium", (120, 180)),
                (100, (5, 10), 80, "details", (100, 160))
            ], 40),
            
            # Phase 3: 512x512 - Details
            (512, [
                (150, (10, 15), 90, "fine_details", (90, 150)),
                (200, (8, 10), 90, "fine_details 2 ", (80, 140)),
                (200, (3, 5), 90, "refinement", (70, 130)),
                (200, (1, 3), 90, "refinement 2 ", (50, 130)),
                (200, (1, 3), 90, "refinement 3 ", (50, 130)),
                (200, (0.5, 1), 80, "refinement 4 ", (50, 120))
                
            ], 50),
            (512, [
                (200, (1, 3), 90, "refinement 3 ", (50, 130)),
                (200, (0.5, 1), 80, "refinement 4 ", (50, 120))
            ], 50)
        ]
        
        self.accumulated_strokes = []
        self.patience = 15
        
    def _preload_brushes(self):
        """Preload brush strokes of different sizes"""
        sizes = [5, 6, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50, 60, 70, 80]
        for size in sizes:
            aspect_ratio = self.brush_original.width / self.brush_original.height
            new_width = int(size * aspect_ratio)
            brush_resized = self.brush_original.resize((new_width, size), Image.LANCZOS)
            self.brush_cache[size] = brush_resized
    
    def _get_brush(self, size: int) -> Image.Image:
        """Get cached brush"""
        if size in self.brush_cache:
            return self.brush_cache[size]
        closest = min(self.brush_cache.keys(), key=lambda x: abs(x - size))
        return self.brush_cache[closest]
    
    def set_resolution(self, resolution: int):
        """Set current working resolution and precompute target features"""
        self.current_resolution = resolution
        self.target = cv2.resize(self.target_original, (resolution, resolution))
        self.target_gray = cv2.cvtColor(self.target, cv2.COLOR_BGR2GRAY)
        self.target_edges = cv2.Canny(self.target_gray, 50, 150)
        
        # Compute gradient direction
        sobelx = cv2.Sobel(self.target_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(self.target_gray, cv2.CV_64F, 0, 1, ksize=3)
        self.gradient_direction = np.arctan2(sobely, sobelx) * 180 / np.pi
        
        # Precompute histograms
        self.target_hist = [cv2.calcHist([self.target], [i], None, [256], [0, 256]) for i in range(3)]
    
    def upscale_strokes(self, scale_factor: float):
        """Upscale all accumulated strokes by scale factor"""
        self.canvas_cache = None
        self.canvas_cache_cv = None
        for i in range(len(self.accumulated_strokes)):
            layer = self.accumulated_strokes[i].copy()
            layer[:, 0] *= scale_factor  # x
            layer[:, 1] *= scale_factor  # y
            layer[:, 5] *= scale_factor  # size
            # Keep color, rotation, alpha the same
            self.accumulated_strokes[i] = layer
    
    def sample_color_from_region(self, x: int, y: int, radius: int = 10) -> Tuple[int, int, int]:
        """Sample average color from region"""
        res = self.current_resolution
        x1 = max(0, int(x - radius))
        y1 = max(0, int(y - radius))
        x2 = min(res, int(x + radius))
        y2 = min(res, int(y + radius))
        
        region = self.target[y1:y2, x1:x2]
        if region.size == 0:
            return (128, 128, 128)
        avg_color = np.mean(region, axis=(0, 1))
        return tuple(map(int, avg_color[::-1]))  # BGR to RGB
    
    def get_gradient_angle(self, x: int, y: int) -> float:
        """Get gradient direction at point"""
        res = self.current_resolution
        x = np.clip(int(x), 0, res - 1)
        y = np.clip(int(y), 0, res - 1)
        return self.gradient_direction[y, x]
    
    def create_smart_individual(self, num_strokes: int, size_range: Tuple[int, int], 
                               layer_type: str, alpha_range: Tuple[int, int]) -> np.ndarray:
        """Create individual with smart initialization"""
        individual = np.zeros((num_strokes, 8))
        res = self.current_resolution
        
        if layer_type == "background":
            grid_size = int(np.sqrt(num_strokes)) + 1
            spacing = res // grid_size
            idx = 0
            
            for i in range(grid_size):
                for j in range(grid_size):
                    if idx >= num_strokes:
                        break
                    x = min(i * spacing + spacing // 2, res - 1)
                    y = min(j * spacing + spacing // 2, res - 1)
                    
                    individual[idx, 0] = x + np.random.randint(-spacing//3, spacing//3 + 1)
                    individual[idx, 1] = y + np.random.randint(-spacing//3, spacing//3 + 1)
                    
                    r, g, b = self.sample_color_from_region(individual[idx, 0], individual[idx, 1], radius=15)
                    individual[idx, 2:5] = [r, g, b]
                    individual[idx, 5] = np.random.randint(size_range[0], size_range[1])
                    
                    angle = self.get_gradient_angle(individual[idx, 0], individual[idx, 1])
                    individual[idx, 6] = (angle + np.random.randint(-20, 20)) % 360
                    individual[idx, 7] = np.random.randint(alpha_range[0], alpha_range[1])
                    idx += 1
                    
        elif layer_type in ["base", "structure", "medium"]:
            for i in range(num_strokes):
                if random.random() < 0.6:
                    edge_points = np.argwhere(self.target_edges > 100)
                    if len(edge_points) > 0:
                        point = edge_points[np.random.randint(len(edge_points))]
                        individual[i, 1], individual[i, 0] = point
                    else:
                        individual[i, 0] = np.random.randint(0, res)
                        individual[i, 1] = np.random.randint(0, res)
                else:
                    individual[i, 0] = np.random.randint(0, res)
                    individual[i, 1] = np.random.randint(0, res)
                
                r, g, b = self.sample_color_from_region(individual[i, 0], individual[i, 1], radius=8)
                individual[i, 2:5] = [r, g, b]
                individual[i, 5] = np.random.randint(size_range[0], size_range[1])
                
                angle = self.get_gradient_angle(individual[i, 0], individual[i, 1])
                individual[i, 6] = (angle + np.random.randint(-30, 30)) % 360
                individual[i, 7] = np.random.randint(alpha_range[0], alpha_range[1])
                
        else:  # details, fine_details, refinement
            individual[:, 0] = np.random.randint(0, res, num_strokes)
            individual[:, 1] = np.random.randint(0, res, num_strokes)
            
            for i in range(num_strokes):
                r, g, b = self.sample_color_from_region(individual[i, 0], individual[i, 1], radius=5)
                individual[i, 2:5] = [r, g, b]
                
                angle = self.get_gradient_angle(individual[i, 0], individual[i, 1])
                individual[i, 6] = (angle + np.random.randint(-45, 45)) % 360
            
            individual[:, 5] = np.random.randint(size_range[0], size_range[1], num_strokes)
            individual[:, 7] = np.random.randint(alpha_range[0], alpha_range[1], num_strokes)
        
        return individual
    
    def render_strokes(self, strokes_list: List[np.ndarray], use_cache: bool = False) -> np.ndarray:
        """Render strokes at current resolution"""
        res = self.current_resolution
        
        if use_cache and self.canvas_cache is not None:
            canvas = self.canvas_cache.copy()
            cached_layers_count = len(self.accumulated_strokes)
            strokes_to_render = strokes_list[cached_layers_count:]
        else:
            canvas = Image.new('RGB', (res, res), (255, 255, 255))
            strokes_to_render = strokes_list
        
        for strokes in strokes_to_render:
            for stroke in strokes:
                x, y, r, g, b, size, rotation, alpha = stroke
                
                # ВАЖНО: проверяем что координаты в пределах текущего разрешения
                if x < 0 or x >= res or y < 0 or y >= res:
                    print(f"WARNING: Stroke outside bounds: ({x}, {y}) for res {res}")
                    continue
                
                x = int(np.clip(x, 0, res - 1))
                y = int(np.clip(y, 0, res - 1))
                
                brush = self._get_brush(int(size))
                brush = brush.rotate(rotation, expand=True, resample=Image.BILINEAR)
                
                colored = Image.new('RGBA', brush.size, (int(r), int(g), int(b), int(alpha)))
                colored.putalpha(brush.split()[3])
                
                paste_x = int(x - brush.width // 2)
                paste_y = int(y - brush.height // 2)
                
                try:
                    canvas.paste(colored, (paste_x, paste_y), colored)
                except Exception as e:
                    print(f"Error pasting stroke at ({x}, {y}): {e}")
        
        return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    
    def fitness(self, current_layer: np.ndarray) -> float:
        """Calculate fitness with MSE, histogram, edge, and SSIM"""
        all_strokes = self.accumulated_strokes + [current_layer]
        rendered = self.render_strokes(all_strokes, use_cache=True)
        
        # MSE
        mse = np.mean((self.target.astype(float) - rendered.astype(float)) ** 2)
        
        # Histogram
        hist_diff = 0
        for i in range(3):
            hist = cv2.calcHist([rendered], [i], None, [256], [0, 256])
            hist_diff += cv2.compareHist(self.target_hist[i], hist, cv2.HISTCMP_CHISQR)
        hist_diff /= 3
        
        # Edge
        rendered_gray = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY)
        rendered_edges = cv2.Canny(rendered_gray, 50, 150)
        edge_diff = np.mean((self.target_edges.astype(float) - rendered_edges.astype(float)) ** 2)

        sobel_t = cv2.Sobel(self.target_gray, cv2.CV_32F, 1, 1)
        sobel_r = cv2.Sobel(rendered_gray, cv2.CV_32F, 1, 1)
        edge_diff = np.mean((sobel_t - sobel_r)**2)

        
        # SSIM
        ssim_score = ssim(self.target_gray, rendered_gray, data_range=255)
        ssim_dissimilarity = (1 - ssim_score) * 10000
        
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
        """Mutate individual"""
        mutated = individual.copy()
        res = self.current_resolution
        
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                gene = random.randint(0, 7)
                
                if gene == 0:  # x
                    mutated[i, 0] = np.clip(mutated[i, 0] + np.random.randint(-20, 20), 0, res - 1)
                elif gene == 1:  # y
                    mutated[i, 1] = np.clip(mutated[i, 1] + np.random.randint(-20, 20), 0, res - 1)
                elif gene in [2, 3, 4]:  # color
                    if random.random() < 0.5:
                        r, g, b = self.sample_color_from_region(mutated[i, 0], mutated[i, 1], radius=8)
                        mutated[i, 2:5] = [r, g, b]
                    else:
                        mutated[i, gene] = np.clip(mutated[i, gene] + np.random.randint(-20, 20), 0, 255)
                elif gene == 5:  # size
                    mutated[i, 5] = np.clip(mutated[i, 5] + np.random.randint(-3, 3), size_range[0], size_range[1])
                elif gene == 6:  # rotation
                    if random.random() < 0.3:
                        angle = self.get_gradient_angle(mutated[i, 0], mutated[i, 1])
                        mutated[i, 6] = (angle + np.random.randint(-30, 30)) % 360
                    else:
                        mutated[i, 6] = (mutated[i, 6] + np.random.randint(-45, 45)) % 360
                elif gene == 7:  # alpha
                    mutated[i, 7] = np.clip(mutated[i, 7] + np.random.randint(-20, 20), alpha_range[0], alpha_range[1])
        
        return mutated
    
    def evolve_layer(self, layer_num: int, num_strokes: int, size_range: Tuple[int, int], 
                     generations: int, layer_type: str, alpha_range: Tuple[int, int]) -> Tuple:
        """Evolve one layer with early stopping"""
        print(f"\n=== Layer {layer_num} ({layer_type}): {num_strokes} strokes, "
              f"size {size_range}, max {generations} generations, res {self.current_resolution} ===")
        
        population = [self.create_smart_individual(num_strokes, size_range, layer_type, alpha_range) 
                     for _ in range(self.population_size)]
        
        avg_fitness_history = []
        max_fitness_history = []
        best_fitness = float('inf')
        patience_counter = 0
        
        for gen in range(generations):
            fitness_scores = [self.fitness(ind) for ind in population]
            
            avg_fitness = np.mean(fitness_scores)
            min_fitness = np.min(fitness_scores)
            avg_fitness_history.append(avg_fitness)
            max_fitness_history.append(min_fitness)
            
            if min_fitness < best_fitness - 1.0:
                best_fitness = min_fitness
                patience_counter = 0
            else:
                patience_counter += 1
            
            if gen % 10 == 0:
                print(f"Gen {gen}: Avg={avg_fitness:.2f}, Best={min_fitness:.2f}, Patience={patience_counter}/{self.patience}")
            
            if patience_counter >= self.patience:
                print(f"Early stopping at gen {gen}")
                break
            
            sorted_indices = np.argsort(fitness_scores)
            sorted_population = [population[i] for i in sorted_indices]
            
            new_population = sorted_population[:self.elitism_count]
            
            while len(new_population) < self.population_size:
                parent1 = sorted_population[random.randint(0, self.population_size // 2)]
                parent2 = sorted_population[random.randint(0, self.population_size // 2)]
                
                child1, child2 = self.crossover(parent1, parent2)
                
                adaptive_rate = self.mutation_rate * (1 - 0.5 * gen / generations)
                child1 = self.mutate(child1, size_range, alpha_range, adaptive_rate)
                child2 = self.mutate(child2, size_range, alpha_range, adaptive_rate)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        fitness_scores = [self.fitness(ind) for ind in population]
        best_idx = np.argmin(fitness_scores)
        
        print(f"Layer {layer_num} complete. Best fitness: {fitness_scores[best_idx]:.2f}")

        best_layer = population[best_idx]
    
        # Обновить кеш после добавления layer
        self.accumulated_strokes.append(best_layer)
        rendered = self.render_strokes(self.accumulated_strokes, use_cache=False)
        self.canvas_cache = Image.fromarray(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
        self.canvas_cache_cv = rendered
        
        
        return population[best_idx], avg_fitness_history, max_fitness_history
    
    def greedy_refinement(self, iterations: int, region_size: int):
        """Greedy refinement: add strokes to worst error regions"""
        print(f"\n=== Greedy Refinement: {iterations} iterations, region {region_size}x{region_size} ===")
        
        res = self.current_resolution
        
        for iteration in range(iterations):
            # Render current state
            rendered = self.render_strokes(self.accumulated_strokes)
            
            # Compute error map
            error_map = np.mean(np.abs(self.target.astype(float) - rendered.astype(float)), axis=2)
            
            # Find worst regions
            num_regions = 5
            worst_regions = []
            
            for _ in range(num_regions):
                # Apply max pooling to find worst region
                max_val = 0
                max_pos = (0, 0)
                
                for y in range(0, res - region_size, region_size // 2):
                    for x in range(0, res - region_size, region_size // 2):
                        region_error = np.mean(error_map[y:y+region_size, x:x+region_size])
                        if region_error > max_val:
                            max_val = region_error
                            max_pos = (x, y)
                
                worst_regions.append(max_pos)
                # Zero out this region to find next worst
                error_map[max_pos[1]:max_pos[1]+region_size, max_pos[0]:max_pos[0]+region_size] = 0
            
            # Add strokes to worst regions
            new_strokes = []
            for x, y in worst_regions:
                # Add 2-3 strokes per region
                for _ in range(random.randint(2, 3)):
                    stroke_x = x + region_size // 2 + np.random.randint(-region_size//4, region_size//4)
                    stroke_y = y + region_size // 2 + np.random.randint(-region_size//4, region_size//4)
                    stroke_x = np.clip(stroke_x, 0, res - 1)
                    stroke_y = np.clip(stroke_y, 0, res - 1)
                    
                    r, g, b = self.sample_color_from_region(stroke_x, stroke_y, radius=5)
                    size = region_size // 3
                    angle = self.get_gradient_angle(stroke_x, stroke_y)
                    alpha = np.random.randint(100, 180)
                    
                    new_strokes.append([stroke_x, stroke_y, r, g, b, size, angle, alpha])
            
            if new_strokes:
                self.accumulated_strokes.append(np.array(new_strokes))
            
            if iteration % 10 == 0:
                current_fitness = self.fitness(np.zeros((1, 8)))  # Dummy, just to get fitness
                print(f"Greedy iter {iteration}/{iterations}: Fitness={current_fitness:.2f}")

        rendered = self.render_strokes(self.accumulated_strokes, use_cache=False)
        self.canvas_cache = Image.fromarray(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
        self.canvas_cache_cv = rendered
    
    def run(self, run_number: int):
        """Run multi-resolution EA with greedy refinement"""

        self.run_folder = f"run_{7}"
        os.makedirs(self.run_folder, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Starting Multi-Resolution EA with Greedy Refinement - Run {run_number}")
        print(f"{'='*80}")
        
        start_time = time.time()
        self.accumulated_strokes = []
        
        all_avg_histories = []
        all_max_histories = []
        
        # Multi-resolution phases
        for phase_num, (resolution, layers_config, greedy_iters) in enumerate(self.phases, 1):
            print(f"\n{'='*80}")
            print(f"PHASE {phase_num}: Resolution {resolution}x{resolution}")
            print(f"{'='*80}")

            # Set current resolution
            self.set_resolution(resolution)
            
            # Upscale strokes from previous phase
            if phase_num > 1:
                prev_resolution = self.phases[phase_num - 2][0]
                scale_factor = resolution / prev_resolution
                print(f"Upscaling strokes by {scale_factor}x")
                self.upscale_strokes(scale_factor)
            

            if len(self.accumulated_strokes) > 0:
                rendered = self.render_strokes(self.accumulated_strokes, use_cache=False)
                self.canvas_cache = Image.fromarray(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
                self.canvas_cache_cv = rendered
            else:
                self.canvas_cache = None
                self.canvas_cache_cv = None
            
            # Evolve layers at this resolution
            for layer_num, (num_strokes, size_range, generations, layer_type, alpha_range) in enumerate(layers_config, 1):
                global_layer_num = sum(len(p[1]) for p in self.phases[:phase_num-1]) + layer_num
                
                best_layer, avg_hist, max_hist = self.evolve_layer(
                    global_layer_num, num_strokes, size_range, generations, layer_type, alpha_range
                )
                
                # self.accumulated_strokes.append(best_layer)
                all_avg_histories.append(avg_hist)
                all_max_histories.append(max_hist)
            
            # Greedy refinement at this resolution
            region_size = max(8, resolution // 16)
            self.greedy_refinement(greedy_iters, region_size)
            
            # Save intermediate result
            intermediate = self.render_strokes(self.accumulated_strokes)
            intermediate_resized = cv2.resize(intermediate, (512, 512))
            cv2.imwrite(f"{self.run_folder}/temp_phase{phase_num}.jpg", intermediate_resized)
        
        elapsed_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"Total time: {elapsed_time/60:.2f} minutes")
        print(f"{'='*80}")

        print(f"\nGenerating final results at resolution {self.current_resolution}")
        print(f"Total accumulated layers: {len(self.accumulated_strokes)}")

        # Generate final variations
        final_results = []
        
        # Best result
        final_results.append(self.render_strokes(self.accumulated_strokes, use_cache=False))
        
        # 4 slight variations
        for i in range(4):
            mutated_strokes = []
            for layer in self.accumulated_strokes:
                size_range = (int(np.min(layer[:, 5])), int(np.max(layer[:, 5])))
                alpha_range = (int(np.min(layer[:, 7])), int(np.max(layer[:, 7])))
                mutated = self.mutate(layer.copy(), size_range, alpha_range, 0.03)
                mutated_strokes.append(mutated)
            final_results.append(self.render_strokes(mutated_strokes))
        
        # Save results
        for i, result in enumerate(final_results, 1):
            output_path = f"{self.run_folder}/{self.output_prefix}Output{self.image_index}_{i}.jpg"
            cv2.imwrite(output_path, result)
            print(f"Saved: {output_path}")
        
        return elapsed_time, all_avg_histories, all_max_histories


# Main execution
if __name__ == "__main__":
    INPUT = "input"
    BRUSH_STROKE = "brush.png"
    OUTPUT_PREFIX = "NameSurname"
    NUM_RUNS = 1
    
    for run in range(1, NUM_RUNS + 1):
        ea = BrushStrokeEA(f"{INPUT}{7}.jpg", BRUSH_STROKE, OUTPUT_PREFIX)
        elapsed, avg_histories, max_histories = ea.run(run)
        
        print(f"\n{'='*80}")
        print(f"Run {run} completed in {elapsed/60:.2f} minutes")
        print(f"{'='*80}\n")