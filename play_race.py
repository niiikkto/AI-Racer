import pygame
from stable_baselines3 import PPO
from race_env import CyberRacingEnv
import os

# Настройки
MODELS_DIR = "models/CyberLive"
LOG_DIR = "logs_cyber_live"
SHOW_EVERY_STEPS = 10000  # Каждые 5000 шагов показываем результат
EPISODES_TO_SHOW = 2     # Сколько заездов показывать за раз

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def main():
    print("--- ЗАПУСК ЖИВОГО ОБУЧЕНИЯ ---")
    print(f"Тренировка {SHOW_EVERY_STEPS} шагов -> Демонстрация -> Повтор")

    # 1. Создаем ДВЕ среды
    # env_train: Без графики, работает максимально быстро
    env_train = CyberRacingEnv(render_mode=None)
    
    # env_show: С графикой, для твоих глаз
    env_show = CyberRacingEnv(render_mode="human")

    # 2. Создаем или загружаем модель
    # Если хочешь продолжить обучение, раскомментируй load
    # model = PPO.load("models/CyberLive/650000", env=env_train)
    model = PPO("MlpPolicy", env_train, verbose=0, tensorboard_log=LOG_DIR, device="auto")

    generation = 0
    
    try:
        while True:
            generation += 1
            print(f"\n>>> ГЕНЕРАЦИЯ {generation}: Идет жесткое обучение ({SHOW_EVERY_STEPS} шагов)...")
            
            # --- ФАЗА 1: ТРЕНИРОВКА (Быстрая) ---
            model.learn(total_timesteps=SHOW_EVERY_STEPS, reset_num_timesteps=False)
            model.save(f"{MODELS_DIR}/{generation * SHOW_EVERY_STEPS}")
            
            # --- ФАЗА 2: ДЕМОНСТРАЦИЯ (Красивая) ---
            print(f">>> ГЕНЕРАЦИЯ {generation}: Смотрим, чему научился...")
            
            for ep in range(EPISODES_TO_SHOW):
                obs, _ = env_show.reset()
                done = False
                total_reward = 0
                
                # Пишем номер поколения в заголовок окна (небольшой хак)
                pygame.display.set_caption(f"CyberRace AI - Gen {generation} - Replay {ep+1}")

                while not done:
                    # Спрашиваем у модели действие
                    action, _ = model.predict(obs)
                    
                    # Делаем шаг в красивой среде
                    obs, reward, terminated, truncated, info = env_show.step(action)
                    done = terminated or truncated  # Эпизод завершен если terminated или truncated
                    total_reward += reward
                    
                    # Обработка выхода через крестик окна во время просмотра
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt

                print(f"   Заезд {ep+1}: Награда = {total_reward:.1f}")

    except KeyboardInterrupt:
        print("\nОстановка обучения. Сохраняю модель...")
        model.save(f"{MODELS_DIR}/final_model")
        env_train.close()
        env_show.close()
        print("Готово.")

if __name__ == "__main__":
    main()