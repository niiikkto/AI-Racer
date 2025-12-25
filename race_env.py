import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

class CyberRacingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super(CyberRacingEnv, self).__init__()
        
        self.window_width = 1000
        self.window_height = 800
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Действия: [Руль (-1..1), Газ (-1..1)]
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        
        # Наблюдение: 7 лучей + скорость + угол руля
        self.observation_space = spaces.Box(low=0, high=1, shape=(9,), dtype=np.float32)

        # Генерация трассы (маска коллизий)
        self.track_surface = pygame.Surface((self.window_width, self.window_height))
        self.checkpoints = [] # Список прямоугольников-чекпоинтов
        self._generate_map()
        
        # Данные машины
        self.car_pos = np.array([100.0, 150.0]) # Старт
        self.car_angle = 0.0
        self.car_speed = 0.0
        self.current_checkpoint = 0
        self.laps = 0
        self.prev_pos = np.array([100.0, 150.0]) # Предыдущая позиция для проверки направления

    def _generate_map(self):
        # Инициализируем pygame если еще не инициализирован (для работы с Surface)
        if not pygame.get_init():
            pygame.init()
        
        # Рисуем карту: Черный фон (смерть), Серый асфальт (жизнь)
        self.track_surface.fill((0, 0, 0))
        
        # Координаты точек трассы (сложная петля)
        points = [
            (100, 150), (400, 100), (800, 150), # Верхняя прямая
            (900, 400), (800, 700), # Правый поворот
            (500, 700), (300, 500), # Петля в центре
            (600, 400), (500, 250),
            (200, 300), (100, 600), # Левый низ
            (50, 400) # Возврат
        ]
        
        # Сглаживаем углы для маски коллизий (используем те же точки, что и для отрисовки)
        smoothed_points = self._smooth_track_points(points, smoothness=30)
        
        # Рисуем широкую дорогу (белый цвет для маски) с закругленными углами
        pygame.draw.lines(self.track_surface, (255, 255, 255), True, smoothed_points, 140)
        
        # Генерируем чекпоинты вдоль линии
        self.checkpoints = []
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            # Ставим чекпоинт посередине сегмента
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            # Создаем Rect размером 140x140 вокруг точки
            rect = pygame.Rect(mid_x - 70, mid_y - 70, 140, 140)
            self.checkpoints.append(rect)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.car_pos = np.array([100.0, 150.0]) # Координаты первого чекпоинта
        self.car_angle = 0.0 # Радианы
        self.car_speed = 0.0
        self.current_checkpoint = 0
        self.laps = 0
        self.steps = 0
        self.prev_pos = np.array([100.0, 150.0]) # Предыдущая позиция для проверки направления
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), {}

    def _get_obs(self):
        # 1. Raycasting (лидары)
        rays = np.linspace(-math.pi/2, math.pi/2, 7) # 7 лучей на 180 градусов
        readings = []
        
        # Оптимизированный raycast: проверяем пиксели на track_surface
        # Белый (255,255,255) = Дорога, Черный (0,0,0) = Стена
        
        for ray_angle in rays:
            angle = self.car_angle + ray_angle
            dist = 0
            max_dist = 200 # Дальность видения
            
            # Шагаем по лучу
            for d in range(5, max_dist, 10): # Шаг 10 пикселей для скорости
                chk_x = int(self.car_pos[0] + math.cos(angle) * d)
                chk_y = int(self.car_pos[1] + math.sin(angle) * d)
                
                # Если вышли за экран или попали в черное
                if (chk_x < 0 or chk_x >= self.window_width or 
                    chk_y < 0 or chk_y >= self.window_height):
                    dist = d
                    break
                
                # Проверка цвета пикселя (0 = черный)
                try:
                    color = self.track_surface.get_at((chk_x, chk_y))[0]
                    if color < 50: # Стена
                        dist = d
                        break
                except:
                    dist = d
                    break
            else:
                dist = max_dist
                
            readings.append(dist / max_dist)

        # Добавляем скорость и данные
        final_obs = np.concatenate([
            readings, 
            [self.car_speed / 15.0],
            [math.sin(self.car_angle)]
        ], dtype=np.float32)
        
        return final_obs

    def step(self, action):
        steering = action[0] 
        throttle = action[1]
        
        # Физика
        self.car_angle += steering * 0.15 # Чувствительность руля
        self.car_speed += throttle * 0.5
        
        # Трение и инерция
        if self.car_speed > 15: self.car_speed = 15
        if self.car_speed < -5: self.car_speed = -5
        self.car_speed *= 0.95 # Трение асфальта
        
        # Движение
        self.car_pos[0] += math.cos(self.car_angle) * self.car_speed
        self.car_pos[1] += math.sin(self.car_angle) * self.car_speed
        
        self.steps += 1
        reward = 0
        terminated = False
        
        # 0. Проверка неправильного направления (разворот назад)
        # Проверяем, если машина едет назад (отрицательная скорость) или удаляется от следующего чекпоинта
        if abs(self.car_speed) > 0.5:  # Только если машина движется
            # Получаем следующий чекпоинт
            next_cp_idx = (self.current_checkpoint + 1) % len(self.checkpoints)
            next_cp_rect = self.checkpoints[next_cp_idx]
            next_cp_center = np.array([next_cp_rect.centerx, next_cp_rect.centery])
            
            # Вектор от машины к следующему чекпоинту
            to_checkpoint = next_cp_center - self.car_pos
            dist_to_checkpoint = np.linalg.norm(to_checkpoint)
            
            # Вектор движения машины (направление скорости)
            if self.car_speed < 0:  # Едет назад
                # Если скорость отрицательная - это явный разворот
                reward = -100  # Большой штраф
                terminated = True
                # Сброс на старт
                self.car_pos = np.array([100.0, 150.0])
                self.car_angle = 0.0
                self.car_speed = 0.0
                self.current_checkpoint = 0
                self.prev_pos = np.array([100.0, 150.0])
                
                # Возвращаемся сразу, чтобы не обрабатывать другие проверки
                if self.render_mode == "human":
                    self._render_frame()
                return self._get_obs(), reward, terminated, False, {}
            else:
                # Проверяем, удаляется ли машина от следующего чекпоинта
                movement = self.car_pos - self.prev_pos
                if np.linalg.norm(movement) > 0.1:  # Если машина сдвинулась
                    # Нормализуем векторы
                    movement_norm = movement / np.linalg.norm(movement)
                    to_checkpoint_norm = to_checkpoint / dist_to_checkpoint if dist_to_checkpoint > 0 else to_checkpoint
                    
                    # Скалярное произведение: если < 0, значит движется в противоположную сторону
                    dot_product = np.dot(movement_norm, to_checkpoint_norm)
                    
                    # Если машина удаляется от чекпоинта (и расстояние достаточно большое)
                    if dot_product < -0.3 and dist_to_checkpoint > 100:
                        reward = -100  # Большой штраф
                        terminated = True
                        # Сброс на старт
                        self.car_pos = np.array([100.0, 150.0])
                        self.car_angle = 0.0
                        self.car_speed = 0.0
                        self.current_checkpoint = 0
                        self.prev_pos = np.array([100.0, 150.0])
                        
                        # Возвращаемся сразу, чтобы не обрабатывать другие проверки
                        if self.render_mode == "human":
                            self._render_frame()
                        return self._get_obs(), reward, terminated, False, {}
        
        # Сохраняем текущую позицию для следующего шага
        self.prev_pos = self.car_pos.copy()
        
        # 1. Проверка столкновения (Стена)
        # Просто смотрим цвет пикселя под машиной
        try:
            cx, cy = int(self.car_pos[0]), int(self.car_pos[1])
            if self.track_surface.get_at((cx, cy))[0] < 50:
                reward = -50 # БОЛЬШОЙ ШТРАФ
                terminated = True
        except IndexError:
            reward = -50
            terminated = True
            
        # 2. Проверка Чекпоинтов (НАГРАДА ЗА ПРОГРЕСС)
        # Получаем следующий чекпоинт
        next_cp_idx = (self.current_checkpoint + 1) % len(self.checkpoints)
        next_cp_rect = self.checkpoints[next_cp_idx]
        
        # Создаем rect машины
        car_rect = pygame.Rect(self.car_pos[0]-10, self.car_pos[1]-10, 20, 20)
        
        if car_rect.colliderect(next_cp_rect):
            reward += 20 # УРА, ЧЕКПОИНТ!
            self.current_checkpoint = next_cp_idx
            
            # Если это был последний чекпоинт - значит КРУГ!
            if self.current_checkpoint == 0:
                reward += 1000 # ФИНИШ!
                self.laps += 1
                terminated = True # Заканчиваем эпизод на победе (или можно продолжать)
        
        # Маленький штраф за время, чтобы не стоял
        reward -= 0.05
        
        if self.steps > 1500: # Тайм-аут
            terminated = True
            
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), reward, terminated, False, {}

    def _smooth_track_points(self, points, smoothness=30):
        """Добавляет промежуточные точки для закругления углов трассы (Catmull-Rom сплайны)"""
        if len(points) < 3:
            return points
        
        smoothed = []
        n = len(points)
        
        # Используем Catmull-Rom сплайны для гарантированной непрерывности
        for i in range(n):
            p0 = points[(i - 1) % n]  # Предыдущая точка
            p1 = points[i]  # Текущая точка
            p2 = points[(i + 1) % n]  # Следующая точка
            p3 = points[(i + 2) % n]  # Следующая следующая точка
            
            # Добавляем промежуточные точки между p1 и p2
            # Пропускаем первую точку сегмента, чтобы избежать дублирования
            start_j = 0 if i == 0 else 1
            for j in range(start_j, smoothness + 1):
                t = j / smoothness
                # Catmull-Rom сплайн - гарантирует прохождение через p1 и p2
                t2 = t * t
                t3 = t2 * t
                
                x = 0.5 * (2*p1[0] + (-p0[0] + p2[0])*t + 
                          (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0])*t2 + 
                          (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0])*t3)
                y = 0.5 * (2*p1[1] + (-p0[1] + p2[1])*t + 
                          (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1])*t2 + 
                          (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1])*t3)
                smoothed.append((int(x), int(y)))
        
        return smoothed

    def _render_frame(self):
        if self.window is None:
            if not pygame.get_init():
                pygame.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Consolas", 20)

        # 1. Отрисовка фона
        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((10, 10, 20)) # Темно-синий фон (Стена/Смерть)
        
        # 2. Отрисовка Дороги (Асфальта)
        # Мы просто берем маску track_surface и используем её как трафарет
        # Создаем белую поверхность (цвет дороги)
        road_layer = pygame.Surface((self.window_width, self.window_height))
        road_layer.fill((40, 40, 50)) # Серый асфальт
        
        # Используем BLEND_RGBA_MULT, чтобы наложить маску
        # track_surface у нас черно-белый (белый=дорога, черный=фон)
        # Но для простоты: просто отрисуем track_surface поверх, но с нужным цветом
        # САМЫЙ ПРОСТОЙ СПОСОБ: Заново нарисовать линии дороги серым цветом
        # (Это дублирование логики, но это 100% работает без магии пикселей)
        
        points = [
            (100, 150), (400, 100), (800, 150), 
            (900, 400), (800, 700), 
            (500, 700), (300, 500), 
            (600, 400), (500, 250),
            (200, 300), (100, 600), 
            (50, 400)
        ]
        # Сглаживаем углы трассы (больше smoothness для более плавных углов)
        smoothed_points = self._smooth_track_points(points, smoothness=30)
        
        # Рисуем контур трассы (белый контур снаружи)
        pygame.draw.lines(canvas, (255, 255, 255), True, smoothed_points, 150)  # Белый контур
        
        # Рисуем саму трассу (асфальт)
        pygame.draw.lines(canvas, (40, 40, 50), True, smoothed_points, 140)  # Серый Асфальт
        pygame.draw.aalines(canvas, (40, 40, 50), True, smoothed_points)  # Сглаженные края для антиалиасинга
        
        # Рисуем одну белую полоску разметки по центру
        pygame.draw.lines(canvas, (255, 255, 255), True, smoothed_points, 2)  # Белая разметка по центру

        # 3. Рисуем Чекпоинты
        for i, cp in enumerate(self.checkpoints):
            # Подсвечиваем только следующий чекпоинт
            if i == (self.current_checkpoint + 1) % len(self.checkpoints):
                pygame.draw.rect(canvas, (0, 255, 0), cp, 1) # Зеленая рамка
            # Остальные не рисуем, чтобы не засорять экран

        # 4. Рисуем машину
        car_surf = pygame.Surface((30, 16), pygame.SRCALPHA)
        pygame.draw.rect(car_surf, (255, 50, 50), (0, 0, 30, 16), border_radius=4)  # Красный цвет
        pygame.draw.rect(car_surf, (255, 255, 255), (5, 4, 10, 8)) 
        
        rotated_car = pygame.transform.rotate(car_surf, -math.degrees(self.car_angle))
        rect = rotated_car.get_rect(center=(int(self.car_pos[0]), int(self.car_pos[1])))
        
        # Свечение (красное)
        glow = pygame.transform.scale(rotated_car, (rect.width+10, rect.height+10))
        glow.fill((100, 0, 0, 50), special_flags=pygame.BLEND_RGBA_ADD)
        canvas.blit(glow, (rect.x-5, rect.y-5))
        
        canvas.blit(rotated_car, rect)

        # 5. Лидары (Лучи)
        obs = self._get_obs()
        rays = np.linspace(-math.pi/2, math.pi/2, 7)
        for i, ray_angle in enumerate(rays):
            # Берем дистанцию из obs (первые 7 элементов)
            dist_val = obs[i] 
            
            angle = self.car_angle + ray_angle
            dist = dist_val * 200
            
            end_x = self.car_pos[0] + math.cos(angle) * dist
            end_y = self.car_pos[1] + math.sin(angle) * dist
            
            # Цвет: Красный если близко (<0.2), иначе Зеленый
            color = (255, 50, 50) if dist_val < 0.2 else (50, 255, 50)
            
            pygame.draw.line(canvas, color, self.car_pos, (end_x, end_y), 1)
            pygame.draw.circle(canvas, color, (int(end_x), int(end_y)), 3)

        # 6. HUD
        ui_text = [
            f"SPEED: {self.car_speed:.1f}",
            f"CHECKPOINT: {self.current_checkpoint}",
            f"LAPS: {self.laps}",
        ]
        for i, line in enumerate(ui_text):
            text = self.font.render(line, True, (0, 255, 0))
            canvas.blit(text, (20, 20 + i * 25))

        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])


    def close(self):
        if self.window is not None:
            pygame.quit()