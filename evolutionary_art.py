import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
import time
import random
from typing import List, Tuple
import os
from skimage.metrics import structural_similarity as ssim

from Statistics import Statistics

class BrushStrokeEA:
    def __init__(self, target_path: str, brush_path: str, output_prefix: str):
        """
        Main class for brush stroke painting using evolutionary algorithm
        """
        # Load and resize target image to 512x512
        self.target_original = cv2.imread(target_path)
        self.target_original = cv2.resize(self.target_original, (512, 512))
        
        # Current resolution data (updated per phase)
        self.target = None
        self.target_gray = None
        self.target_edges = None
        self.target_hist = None
        self.gradient_direction = None
        self.current_resolution = None

        self.run_folder = None
        
        # Load brush image
        self.brush_original = Image.open(brush_path).convert('RGBA')
        self.brush_cache = {}
        self._preload_brushes()
        self.canvas_cache = None  # PIL canvas
        self.canvas_cache_cv = None  # OpenCV canvas for fitness
        
        self.output_prefix = output_prefix
        self.image_index = os.path.basename(target_path).replace('input', '').replace('.jpg', '')
        
        # EA settings
        self.population_size = 25
        self.elitism_count = 5
        self.crossover_rate = 0.8
        self.mutation_rate = 0.3
        
        # Fitness weights
        self.w_mse = 0.3
        self.w_hist = 0.35
        self.w_edge = 0.15
        self.w_ssim = 0.2
        
        # Phases: resolution, layers, greedy steps
        self.phases = [
            # Phase 1: low resolution - basic shape and color
            (128, [
                (12, (10, 18), 80, "background", (200, 255)),
                (10, (12, 20), 50, "base", (160, 220)),
                (60, (6, 12), 60, "structure", (130, 190))
            ], 0),
            
            # Phase 2: medium resolution - add structure
            (256, [
                (100, (20, 30), 80, "medium", (120, 180)),
                (100, (15, 20), 80, "details", (100, 160)),
                (100, (5, 15), 80, "details", (100, 160))
            ], 0),
            
            # Phase 3: full resolution - fine details
            (512, [
                (100, (10, 18), 80, "details", (90, 150)),
                (200, (8, 10), 90, "fine_details 2 ", (80, 140)),
                (200, (3, 5), 90, "refinement", (70, 130)),
                (200, (1, 3), 90, "refinement 2 ", (50, 130)),
                (200, (1, 3), 100, "refinement 3 ", (50, 130)),
                (200, (0.5, 1), 100, "refinement 4 ", (50, 130)),
                (200, (0.5, 1), 150, "refinement 4 ", (50, 120))
            ], 0)
        ]
        
        self.accumulated_strokes = []
        self.stroke_resolutions = []
        self.patience = 15
        
    def _preload_brushes(self):
        """Preload resized brushes for speed"""
        sizes = [5, 6, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50, 60, 70, 80]
        for size in sizes:
            aspect_ratio = self.brush_original.width / self.brush_original.height
            new_width = int(size * aspect_ratio)
            brush_resized = self.brush_original.resize((new_width, size), Image.LANCZOS)
            self.brush_cache[size] = brush_resized
    
    def _get_brush(self, size: int) -> Image.Image:
        """Return cached or closest brush"""
        if size in self.brush_cache:
            return self.brush_cache[size]
        closest = min(self.brush_cache.keys(), key=lambda x: abs(x - size))
        return self.brush_cache[closest]
    
    def set_resolution(self, resolution: int):
        """Prepare target image and features for current resolution"""
        self.current_resolution = resolution
        self.target = cv2.resize(self.target_original, (resolution, resolution))
        self.target_gray = cv2.cvtColor(self.target, cv2.COLOR_BGR2GRAY)
        self.target_edges = cv2.Canny(self.target_gray, 50, 150)
        
        # Gradient direction for stroke orientation
        sobelx = cv2.Sobel(self.target_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(self.target_gray, cv2.CV_64F, 0, 1, ksize=3)
        self.gradient_direction = np.arctan2(sobely, sobelx) * 180 / np.pi
        
        # Color histograms
        self.target_hist = [cv2.calcHist([self.target], [i], None, [256], [0, 256]) for i in range(3)]
    
    def upscale_strokes(self, target_resolution: int):
        """Scale all previous strokes to new resolution"""
        self.canvas_cache = None
        self.canvas_cache_cv = None
        
        for i in range(len(self.accumulated_strokes)):
            layer = self.accumulated_strokes[i].copy()
            original_resolution = self.stroke_resolutions[i]
            scale_factor = target_resolution / original_resolution 
            
            layer[:, 0] *= scale_factor  # x
            layer[:, 1] *= scale_factor  # y
            layer[:, 5] *= scale_factor  # size
            
            self.accumulated_strokes[i] = layer
            self.stroke_resolutions[i] = target_resolution
    
    def sample_color_from_region(self, x: int, y: int, radius: int = 10) -> Tuple[int, int, int]:
        """Get average color around point"""
        res = self.current_resolution
        x1 = max(0, int(x - radius))
        y1 = max(0, int(y - radius))
        x2 = min(res, int(x + radius))
        y2 = min(res, int(y + radius))
        
        region = self.target[y1:y2, x1:x2]
        if region.size == 0:
            return (128, 128, 128)
        avg_color = np.mean(region, axis=(0, 1))
        return tuple(map(int, avg_color[::-1]))  # BGR -> RGB
    
    def get_gradient_angle(self, x: int, y: int) -> float:
        """Get edge direction at point"""
        res = self.current_resolution
        x = np.clip(int(x), 0, res - 1)
        y = np.clip(int(y), 0, res - 1)
        return self.gradient_direction[y, x]
    
    def create_smart_individual(self, num_strokes: int, size_range: Tuple[int, int], 
                               layer_type: str, alpha_range: Tuple[int, int]) -> np.ndarray:
        """Create one solution with smart placement"""
        individual = np.zeros((num_strokes, 8))
        res = self.current_resolution
        
        if layer_type == "background":
            # Grid placement for background
            grid_size = int(np.sqrt(num_strokes)) + 1
            spacing = res // grid_size
            idx = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    if idx >= num_strokes: break
                    x = i * spacing + spacing // 2
                    y = j * spacing + spacing // 2
                    individual[idx, 0:2] = [x, y]
                    individual[idx, 2:5] = self.sample_color_from_region(x, y, 15)
                    individual[idx, 5] = np.random.randint(*size_range)
                    individual[idx, 6] = self.get_gradient_angle(x, y) + np.random.randint(-20, 20)
                    individual[idx, 7] = np.random.randint(*alpha_range)
                    idx += 1
                    
        elif layer_type in ["base", "structure", "medium"]:
            # Place most strokes on edges
            for i in range(num_strokes):
                if random.random() < 0.6 and len(np.argwhere(self.target_edges > 100)) > 0:
                    y, x = random.choice(np.argwhere(self.target_edges > 100))
                    individual[i, 0:2] = [x, y]
                else:
                    individual[i, 0:2] = [np.random.randint(0, res), np.random.randint(0, res)]
                individual[i, 2:5] = self.sample_color_from_region(individual[i, 0], individual[i, 1], 8)
                individual[i, 5] = np.random.randint(*size_range)
                individual[i, 6] = self.get_gradient_angle(individual[i, 0], individual[i, 1]) + np.random.randint(-30, 30)
                individual[i, 7] = np.random.randint(*alpha_range)
                
        else:  # details and refinement
            # Random placement with local color
            individual[:, 0:2] = np.random.randint(0, res, (num_strokes, 2))
            for i in range(num_strokes):
                individual[i, 2:5] = self.sample_color_from_region(individual[i, 0], individual[i, 1], 5)
                individual[i, 6] = self.get_gradient_angle(individual[i, 0], individual[i, 1]) + np.random.randint(-45, 45)
            individual[:, 5] = np.random.randint(*size_range, num_strokes)
            individual[:, 7] = np.random.randint(*alpha_range, num_strokes)
        
        individual[:, 6] %= 360
        return individual
    
    def render_strokes(self, strokes_list: List[np.ndarray], use_cache: bool = False) -> np.ndarray:
        """Draw all strokes on canvas"""
        res = self.current_resolution
        
        if use_cache and self.canvas_cache is not None:
            canvas = self.canvas_cache.copy()
            strokes_to_render = strokes_list[len(self.accumulated_strokes):]
        else:
            canvas = Image.new('RGB', (res, res), (255, 255, 255))
            strokes_to_render = strokes_list
        
        draw = ImageDraw.Draw(canvas)
        
        for layer in strokes_to_render:
            for stroke in layer:
                x, y, r, g, b, size, rotation, alpha = stroke.astype(int)
                if not (0 <= x < res and 0 <= y < res): continue
                
                brush = self._get_brush(size).rotate(rotation, expand=True)
                colored = Image.new('RGBA', brush.size, (r, g, b, alpha))
                colored.putalpha(brush.split()[-1])
                
                pos = (int(x - brush.width // 2), int(y - brush.height // 2))
                canvas.paste(colored, pos, colored)
        
        return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    
    def fitness(self, current_layer: np.ndarray) -> float:
        """Calculate how good the painting is"""
        all_strokes = self.accumulated_strokes + [current_layer]
        rendered = self.render_strokes(all_strokes, use_cache=True)
        
        # Color difference
        mse = np.mean((self.target.astype(float) - rendered.astype(float)) ** 2)
        
        # Color distribution difference
        hist_diff = sum(cv2.compareHist(self.target_hist[i], 
                     cv2.calcHist([rendered], [i], None, [256], [0, 256]), cv2.HISTCMP_CHISQR) 
                     for i in range(3)) / 3
        
        # Edge difference
        rendered_gray = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY)
        sobel_t = cv2.Sobel(self.target_gray, cv2.CV_32F, 1, 1)
        sobel_r = cv2.Sobel(rendered_gray, cv2.CV_32F, 1, 1)
        edge_diff = np.mean((sobel_t - sobel_r)**2)
        
        # Structure similarity
        ssim_score = ssim(self.target_gray, rendered_gray, data_range=255)
        ssim_dissimilarity = (1 - ssim_score) * 10000
        
        return (self.w_mse * mse + 
                self.w_hist * hist_diff + 
                self.w_edge * edge_diff + 
                self.w_ssim * ssim_dissimilarity)
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Mix two solutions"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        point = random.randint(1, len(parent1)-1)
        return np.vstack((parent1[:point], parent2[point:])), np.vstack((parent2[:point], parent1[point:]))
    
    def mutate(self, individual: np.ndarray, size_range: Tuple[int, int], 
               alpha_range: Tuple[int, int], mutation_rate: float) -> np.ndarray:
        """Randomly change parts of solution"""
        mut = individual.copy()
        res = self.current_resolution
        
        for i in range(len(mut)):
            if random.random() < mutation_rate:
                gene = random.randint(0, 7)
                if gene == 0: mut[i,0] = np.clip(mut[i,0] + random.randint(-20,20), 0, res-1)
                if gene == 1: mut[i,1] = np.clip(mut[i,1] + random.randint(-20,20), 0, res-1)
                if gene in [2,3,4]:
                    if random.random() < 0.5:
                        mut[i,2:5] = self.sample_color_from_region(mut[i,0], mut[i,1], 8)
                    else:
                        mut[i,gene] = np.clip(mut[i,gene] + random.randint(-20,20), 0, 255)
                if gene == 5: mut[i,5] = np.clip(mut[i,5] + random.randint(-3,4), *size_range)
                if gene == 6:
                    if random.random() < 0.3:
                        mut[i,6] = self.get_gradient_angle(mut[i,0], mut[i,1]) + random.randint(-30,30)
                    else:
                        mut[i,6] = (mut[i,6] + random.randint(-45,45)) % 360
                if gene == 7: mut[i,7] = np.clip(mut[i,7] + random.randint(-20,20), *alpha_range)
        return mut
    
    def evolve_layer(self, layer_num: int, num_strokes: int, size_range: Tuple[int, int], 
                     generations: int, layer_type: str, alpha_range: Tuple[int, int]):
        """Run evolution for one layer"""
        print(f"\nLayer {layer_num} - {num_strokes} strokes, size {size_range}")
        
        population = [self.create_smart_individual(num_strokes, size_range, layer_type, alpha_range) 
                     for _ in range(self.population_size)]
        
        best_fitness = float('inf')
        no_improve = 0
        
        for gen in range(generations):
            scores = [self.fitness(ind) for ind in population]
            best_now = min(scores)
            
            if best_now < best_fitness - 1.0:
                best_fitness = best_now
                no_improve = 0
            else:
                no_improve += 1
                
            if gen % 10 == 0:
                print(f"  Gen {gen}: best {best_now:.1f} (no improve: {no_improve})")
                
            if no_improve >= self.patience:
                print("  Early stop")
                break
                
            # Keep best
            elites = [population[i] for i in np.argsort(scores)[:self.elitism_count]]
            new_pop = elites[:]
            
            # Create new individuals
            while len(new_pop) < self.population_size:
                p1 = random.choice(population[:self.population_size//2])
                p2 = random.choice(population[:self.population_size//2])
                c1, c2 = self.crossover(p1, p2)
                rate = self.mutation_rate * (1 - 0.5 * gen / generations)
                c1 = self.mutate(c1, size_range, alpha_range, rate)
                c2 = self.mutate(c2, size_range, alpha_range, rate)
                new_pop.extend([c1, c2])
                
            population = new_pop[:self.population_size]
        
        # Save best layer
        scores = [self.fitness(ind) for ind in population]
        best = population[np.argmin(scores)]
        self.accumulated_strokes.append(best)
        self.stroke_resolutions.append(self.current_resolution)
        
        # Update canvas cache
        rendered = self.render_strokes(self.accumulated_strokes, use_cache=False)
        self.canvas_cache = Image.fromarray(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
        self.canvas_cache_cv = rendered
        
        return best, [], [], gen + 1
    
    
    def run(self, run_number: int):
        """Main loop - run all phases"""
        self.run_folder = f"run_{self.image_index}_{run_number}"
        os.makedirs(self.run_folder, exist_ok=True)
        
        start_time = time.time()
        self.accumulated_strokes = []
        self.stroke_resolutions = []
        
        for phase_num, (res, layers, greedy) in enumerate(self.phases, 1):
            print(f"\n=== PHASE {phase_num} - {res}x{res} ===")
            self.set_resolution(res)
            
            if phase_num > 1:
                self.upscale_strokes(res)
            
            # Run each layer in phase
            for layer_idx, config in enumerate(layers, 1):
                num_strokes, size_r, gens, ltype, alpha_r = config
                global_layer = sum(len(p[1]) for p in self.phases[:phase_num-1]) + layer_idx
                self.evolve_layer(global_layer, num_strokes, size_r, gens, ltype, alpha_r)
            
            # Save intermediate result
            img = self.render_strokes(self.accumulated_strokes)
            img = cv2.resize(img, (512, 512))
            cv2.imwrite(f"{self.run_folder}/temp_phase{phase_num}.jpg", img)
        
        # Final images
        final = self.render_strokes(self.accumulated_strokes)
        cv2.imwrite(f"{self.run_folder}/{self.output_prefix}Output{self.image_index}_1.jpg", final)
        print("Done")
        
        return time.time() - start_time, [], [], 0, 0, []

if __name__ == "__main__":
    INPUT = "input"
    BRUSH_STROKE = "brush.png"
    OUTPUT_PREFIX = "EgorNovokreshchenov"
    NUM_RUNS = 3
    
    test_images = [1, 2, 3, 4, 5] 
    
    for img_idx in test_images:
        print(f"\nTESTING IMAGE {img_idx}")
        stats = Statistics(OUTPUT_PREFIX, str(img_idx))
        
        for run in range(1, NUM_RUNS + 1):
            ea = BrushStrokeEA(f"{INPUT}{img_idx}.jpg", BRUSH_STROKE, OUTPUT_PREFIX)
            elapsed, _, _, _, _, _ = ea.run(run)
            stats.add_run(run, elapsed, [], [], 0, 0, [])
            print(f"Run {run} finished in {elapsed/60:.1f} min")
        
        stats.plot_fitness_curves()
        stats.generate_summary_table()
        stats.save_json()